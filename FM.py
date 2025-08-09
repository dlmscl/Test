import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import integrate
from sklearn.preprocessing import StandardScaler
import random
import os

# --- new imports for openTSNE and sparse handling ---
from openTSNE import affinity
from scipy import sparse

# 配置参数
class Config:
    # 随机种子
    seed = 42

    # 数据参数
    input_dim = 64  # 输入维度 (e.g. 8x8图像)
    target_dim = 2  # 目标维度
    dataset_size = 1797  # MNIST digits数据集大小

    # 模型参数
    time_embed_dim = 128
    hidden_dim = 256
    num_blocks = 3

    # t-SNE 参数
    tsne_perplexity = 30  # t-SNE 复杂度参数
    tsne_n_iter = 1000  # t-SNE 迭代次数

    # 训练参数
    epochs = 500
    batch_size = 128
    lr = 1e-3

    # 流匹配参数
    ode_steps = 50  # ODE求解步数
    t_bias_range = (0.0, 1.0)  # 时间步偏置范围

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 相似性损失参数
    _lambda = 1

# 设置随机种子函数
def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Set seed to {seed} for all random sources")

# 时间嵌入层
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.proj(t.view(-1, 1))


# 残差块
class MLPBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.time_embed = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim))
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.block(x + self.time_embed(t_emb))
        return h + self.shortcut(x)


# FlowMatch 模型
class FlowMatchModel(nn.Module):
    def __init__(self, input_dim, target_dim, time_dim=128, hidden_dim=256, num_blocks=3,
                 perplexity=30.0, k_steps=5, lambda_sim=0.1):
        super().__init__()
        self.total_dim = input_dim + target_dim
        self.target_dim = target_dim

        # 时间嵌入
        self.time_embed = TimeEmbedding(time_dim)

        # 输入投影
        self.input_proj = nn.Linear(self.total_dim, hidden_dim)

        # 残差块
        self.blocks = nn.ModuleList([
            MLPBlock(hidden_dim, time_dim) for _ in range(num_blocks)
        ])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim))

        # EMA
        self.ema = None
        self.ema_decay = 0.999

        # t-SNE增强参数
        self.perplexity = perplexity
        self.lambda_sim = lambda_sim  # 相似性损失权重

    def forward(self, zt, t):
        # 时间嵌入
        t_emb = self.time_embed(t)

        # 输入投影
        x = self.input_proj(zt)

        # 残差块
        for block in self.blocks:
            x = block(x, t_emb)

        # 输出目标空间速度
        v_target = self.output_layer(x)

        # 创建完整速度向量 (源部分为零)
        v_full = torch.zeros_like(zt)
        v_full[:, -self.target_dim:] = v_target

        return v_full

    def update_ema(self):
        if self.ema is None:
            self.ema = {k: v.detach().clone() for k, v in self.state_dict().items()}
            return

        with torch.no_grad():
            for param_name, param in self.named_parameters():
                self.ema[param_name] = self.ema_decay * self.ema[param_name] + (1 - self.ema_decay) * param.data

    def ema_parameters(self):
        if self.ema is None:
            return self.parameters()
        return self.ema.values()


# --- helper functions to work with precomputed P from openTSNE ---
def get_P_batch(P_full, idxs):
    """从全局 P（稀疏或稠密）中切出 batch 对应的子矩阵并归一化（返回 numpy ndarray）。
    idxs: torch tensor (B,) of integer indices
    """
    idxs_np = idxs.detach().cpu().numpy()
    if sparse.issparse(P_full):
        P_sub = P_full[idxs_np, :][:, idxs_np].toarray()
    else:
        P_sub = P_full[np.ix_(idxs_np, idxs_np)]
    # 数值保护与归一化
    P_sub = np.maximum(P_sub, 1e-12)
    P_sub = P_sub / P_sub.sum()
    return P_sub


def compute_Q_torch(Y):
    """基于 student-t kernel 计算低维 Q（返回 torch.tensor，sum=1）
    Y: (B, d) tensor
    """
    eps = 1e-12
    dist2 = torch.cdist(Y, Y, p=2.0)
    dist2 = dist2.pow(2)
    inv = 1.0 / (1.0 + dist2)
    inv.fill_diagonal_(0.0)
    Q = inv / (inv.sum() + eps)
    return Q


# 修改后的 path_similarity_loss：使用 precomputed P_full（openTSNE 产生）
def path_similarity_loss(model, x, t_values, P_full, batch_idx, perplexity=None):
    """
    沿降维路径计算相似性损失（使用预计算的 P_full）
    - model: FlowMatchModel
    - x: batch 高维输入 (B, D)
    - t_values: 1D tensor of time samples on [0,1]
    - P_full: full joint-probabilities from openTSNE (numpy ndarray or scipy sparse)
    - batch_idx: torch tensor of indices for this batch in the global dataset
    """
    device = x.device
    # 切出 batch 对应的 P 子矩阵并转换为 torch
    P_batch_np = get_P_batch(P_full, batch_idx)
    P_batch = torch.tensor(P_batch_np, dtype=torch.float32, device=device)

    # 初始化联合空间（source 固定）
    zt = torch.cat([x, torch.zeros(x.size(0), model.target_dim, device=device)], dim=1)
    total_steps = len(t_values) - 1

    path_points = []
    for i in range(total_steps):
        current_t = t_values[i].expand(x.size(0), 1)
        v = model(zt, current_t)
        zt = zt + v * (t_values[i + 1] - t_values[i])
        low_dim = zt[:, -model.target_dim:]
        path_points.append((current_t, low_dim.clone()))

    # 计算 KL(P || Q)（使用 P_batch）
    sim_loss = 0.0
    eps = 1e-12
    for t, low_dim in path_points:
        Q_low = compute_Q_torch(low_dim)
        # KL(P||Q) = sum P * (log P - log Q)
        kl = torch.sum(P_batch * (torch.log(P_batch + eps) - torch.log(Q_low + eps)))
        # 可选的时间加权
        w = torch.exp(5 * (t[0] - 1))
        sim_loss = sim_loss + w * kl

    sim_loss = sim_loss / max(len(path_points), 1)
    return sim_loss


# Heun二阶ODE求解器（保持不变）
def ode_solve(model, x, steps=50):
    """从高维x生成低维z"""
    # 初始化联合空间: [源数据, 目标零填充]
    zt = torch.cat([x, torch.zeros(x.size(0), Config.target_dim, device=x.device)], dim=1)

    # 时间步
    t = torch.linspace(0, 1, steps + 1, device=x.device)
    dt = 1 / steps

    # ODE求解
    for i in range(steps):
        current_t = t[i].expand(x.size(0), 1)

        # 动态时间步调整
        adaptive_dt = dt * (1 + 0.5 * torch.sin(2 * np.pi * current_t))  # 曲率高区域减小步长

        # 计算速度
        v = model(zt, current_t)

        # Heun方法: 预测-校正
        k1 = v
        zt_pred = zt + k1 * adaptive_dt

        # 校正步
        if i < steps - 1:
            next_t = t[i + 1].expand(x.size(0), 1)
            k2 = model(zt_pred, next_t)
            zt = zt + 0.5 * (k1 + k2) * adaptive_dt
        else:
            zt = zt_pred

    # 返回目标维度
    return zt[:, -Config.target_dim:]


# 计算 t-SNE 目标（保留，用作训练中的 z 目标或者可视化参考）
def compute_tsne_targets(data):
    print("Computing t-SNE embeddings for each sample...")

    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 计算 t-SNE（sklearn ，只用于生成 a target embedding for visualization / linear-interp target）
    tsne = TSNE(
        n_components=Config.target_dim,
        perplexity=Config.tsne_perplexity,
        n_iter=Config.tsne_n_iter,
        random_state=42
    )
    tsne_result = tsne.fit_transform(data_scaled)

    # 转换为张量
    tsne_tensor = torch.tensor(tsne_result, dtype=torch.float32)

    print("t-SNE computation completed.")
    return scaler, tsne_tensor


# 训练函数（修改：使用 IndexedDataset、传入 batch indices，修正 zt / target_v 一致性）
def train(model, dataloader, optimizer, epoch, scaler, P_full):
    model.train()
    total_fl_loss, total_sim_loss, total_loss = 0, 0, 0

    for i, (x, _, tsne_target, idxs) in enumerate(dataloader):
        x_np = x.numpy()
        x_scaled = scaler.transform(x_np)
        x = torch.tensor(x_scaled, dtype=torch.float32).to(Config.device)

        z = tsne_target = tsne_target.to(Config.device)

        # 偏置时间步采样 (关键区域过采样)
        if np.random.rand() < 0.7:  # 70%概率在关键区间采样
            t = torch.rand(x.size(0), 1, device=Config.device) * \
                (Config.t_bias_range[1] - Config.t_bias_range[0]) + Config.t_bias_range[0]
        else:
            t = torch.rand(x.size(0), 1, device=Config.device)

        # 线性插值路径（修改：source 固定为 x，低维随 t 累加）
        zt = torch.cat([
            (1 - t) * x,
            t * z
        ], dim=1)

        # 目标速度（source 部分为 0，低维速度为 z）
        target_v = torch.cat([
            -x,
            z
        ], dim=1)

        # 模型预测
        pred_v = model(zt, t)

        # 流匹配损失
        fl_loss = F.mse_loss(pred_v, target_v)

        # 2. 路径相似性损失 (使用预计算的 P_full)
        sim_loss = 0
        # 创建时间步序列 (用于路径积分)
        t_values = torch.linspace(0, 1, Config.ode_steps+1, device=x.device)
        sim_loss = path_similarity_loss(model, x, t_values, P_full, idxs, model.perplexity)

        # 总损失
        loss = fl_loss + model.lambda_sim * sim_loss

        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        # 更新EMA
        model.update_ema()
        # 记录损失
        total_loss += loss.item()
        total_fl_loss += fl_loss.item()
        total_sim_loss += sim_loss.item()

    n_batches = len(dataloader)
    return total_fl_loss / n_batches, total_sim_loss / n_batches, total_loss / n_batches


# 评估与可视化（保持原样，注意 dataset 结构变化）
def visualize(model, test_loader, epoch, scaler):
    model.eval()
    all_data = []
    all_labels = []
    all_embeddings = []

    # 收集数据和标签
    with torch.no_grad():
        for x, y, _, _ in test_loader:
            # 标准化输入数据
            x_np = x.numpy()
            x_scaled = scaler.transform(x_np)
            x = torch.tensor(x_scaled, dtype=torch.float32).to(Config.device)

            all_data.append(x.cpu())
            all_labels.append(y.cpu())

    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)

    # 原始数据PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_data.numpy())

    # t-SNE基线
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(all_data.numpy())

    # FlowMatch降维
    batch_size = 256
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size].to(Config.device)
        embeddings = ode_solve(model, batch, steps=Config.ode_steps)
        all_embeddings.append(embeddings.cpu())

    flowmatch_result = torch.cat(all_embeddings).detach().numpy()

    # 可视化
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=all_labels, cmap='tab10', s=5)
    plt.title('PCA')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_labels, cmap='tab10', s=5)
    plt.title('t-SNE')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.scatter(flowmatch_result[:, 0], flowmatch_result[:, 1], c=all_labels, cmap='tab10', s=5)
    plt.title(f'FlowMatch (Epoch {epoch})')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'./visual/tsne-target-simnew_epoch_{epoch}.png')
    plt.close()


# --- 小的辅助 dataset：返回 idx 以便从 P_full 中切片 ---
from torch.utils.data import Dataset
class IndexedDataset(Dataset):
    def __init__(self, data, labels, tsne_targets):
        self.data = data
        self.labels = labels
        self.tsne_targets = tsne_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.tsne_targets[idx], idx


# 主函数
def main():
    # 设置随机种子
    set_seed(Config.seed)
    # 加载数据集 (Digits数据集)
    digits = load_digits()
    data = digits.data / 16.0  # 归一化到[0,1]
    labels = digits.target

    # 计算 t-SNE 目标（用于训练中作为 z 目标 / 可视化）
    scaler, tsne_targets = compute_tsne_targets(data)

    # ---------- 使用 openTSNE 预计算全局 P ----------
    print("Computing openTSNE affinities (P) ...")
    data_scaled = scaler.transform(data)  # openTSNE 接受 numpy array
    aff = affinity.PerplexityBasedNN(data_scaled, perplexity=Config.tsne_perplexity,
                                     method="approx", n_jobs=8, random_state=Config.seed)
    P_full = aff.P  # scipy.sparse 或 numpy array
    print("Computed P_full: ", type(P_full), getattr(P_full, 'shape', None))

    # 转换为PyTorch数据集（返回 idx）
    dataset = IndexedDataset(torch.tensor(data, dtype=torch.float32),
                             torch.tensor(labels, dtype=torch.long),
                             tsne_targets.clone().detach())
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    # 初始化模型
    model = FlowMatchModel(
        input_dim=Config.input_dim,
        target_dim=Config.target_dim,
        time_dim=Config.time_embed_dim,
        hidden_dim=Config.hidden_dim,
        num_blocks=Config.num_blocks,
        lambda_sim=Config._lambda
    ).to(Config.device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)

    # 训练循环
    for epoch in range(Config.epochs):
        fl_loss, sim_loss, loss = train(model, dataloader, optimizer, epoch, scaler, P_full)
        print(f'Epoch {epoch + 1}/{Config.epochs}, Fl_Loss: {fl_loss:.6f}, Sim_Loss: {sim_loss:.6f}, Loss: {loss:.6f}')

        # 每100个epoch可视化一次
        if (epoch + 1) % 100 == 0 or epoch == 0:
            visualize(model, DataLoader(dataset, batch_size=1024), epoch + 1, scaler)

    print("训练完成!")

    # 最终可视化
    visualize(model, DataLoader(dataset, batch_size=1024), Config.epochs, scaler)


if __name__ == "__main__":
    main()
