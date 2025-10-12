import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log_features(epoch, spe_f, spa_f, save_dir="./feature_logs/", prefix="train", num_samples=2000):
    """
    可视化并保存频谱特征 vs 空间特征的对比图 (低内存版)
    Args:
        epoch: 当前epoch编号
        spe_f: (N, C1, H, W) 频谱特征
        spa_f: (N, C2, H, W) 空间特征
        save_dir: 保存目录
        prefix: "train" 或 "test"
        num_samples: 每次最多采样多少点用于可视化 (避免爆内存)
    """
    ensure_dir(save_dir)

    spe = spe_f.detach().cpu().view(spe_f.size(0), spe_f.size(1), -1)  # (N, C1, L)
    spa = spa_f.detach().cpu().view(spa_f.size(0), spa_f.size(1), -1)  # (N, C2, L)

    spe = spe.mean(dim=0).T  # (L, C1)
    spa = spa.mean(dim=0).T  # (L, C2)

    # 随机采样，防止内存爆炸
    L = spe.size(0)
    num_samples = min(num_samples, L)
    idx = torch.randperm(L)[:num_samples]
    spe = spe[idx]
    spa = spa[idx]

    # 1. Cosine similarity 分布
    spe_norm = spe / (spe.norm(dim=-1, keepdim=True) + 1e-6)
    spa_norm = spa / (spa.norm(dim=-1, keepdim=True) + 1e-6)

    min_dim = min(spe_norm.size(1), spa_norm.size(1))
    cos = (spe_norm[:, :min_dim] * spa_norm[:, :min_dim]).sum(dim=-1).cpu().numpy()

    plt.figure()
    sns.histplot(cos, bins=50, kde=True)
    plt.title("Cosine Similarity Distribution (sampled)")
    plt.xlabel("cosine similarity")
    plt.ylabel("frequency")
    plt.savefig(os.path.join(save_dir, f"{prefix}_epoch{epoch}_cosine.png"))
    plt.close()

    # 2. Cross-correlation 矩阵
    spe_norm = (spe - spe.mean(0)) / (spe.std(0) + 1e-6)
    spa_norm = (spa - spa.mean(0)) / (spa.std(0) + 1e-6)
    cov = torch.matmul(spe_norm.T, spa_norm) / spe_norm.size(0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cov.numpy(), cmap="coolwarm", center=0)
    plt.title("Cross-Correlation Matrix (sampled)")
    plt.savefig(os.path.join(save_dir, f"{prefix}_epoch{epoch}_crosscorr.png"))
    plt.close()

    # 3. 每通道方差对比
    spe_var = spe.var(0).numpy()
    spa_var = spa.var(0).numpy()
    plt.figure()
    plt.plot(spe_var, label="spectral var")
    plt.plot(spa_var, label="spatial var")
    plt.legend()
    plt.title("Per-channel Variance (sampled)")
    plt.savefig(os.path.join(save_dir, f"{prefix}_epoch{epoch}_variance.png"))
    plt.close()

    # 4. t-SNE 可视化
    feat = torch.cat([spe, spa], dim=1).numpy()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    feat_2d = tsne.fit_transform(feat)

    plt.figure(figsize=(6, 6))
    plt.scatter(feat_2d[:spe.size(0), 0], feat_2d[:spe.size(0), 1],
                c="blue", label="spectral", alpha=0.6)
    plt.scatter(feat_2d[:spa.size(0), 0], feat_2d[:spa.size(0), 1],
                c="red", label="spatial", alpha=0.6)
    plt.legend()
    plt.title("t-SNE: Spectral vs Spatial (sampled)")
    plt.savefig(os.path.join(save_dir, f"{prefix}_epoch{epoch}_tsne.png"))
    plt.close()

    print(f"[features_logger] Saved feature comparison at epoch {epoch} → {save_dir}")



def plot_correlation_matrices(spe_f, spa_f, save_path="./feature_logs/correlation_matrice", num_features=50):
    """对比频谱特征和空间特征的相关性矩阵"""

    ensure_dir(save_path)

    # 采样特征以减少计算量
    spe_sample = spe_f[:, :num_features].view(spe_f.size(0), -1).cpu().numpy()
    spa_sample = spa_f[:, :num_features].view(spa_f.size(0), -1).cpu().numpy()
    
    # 计算相关性矩阵
    spe_corr = np.corrcoef(spe_sample.T)
    spa_corr = np.corrcoef(spa_sample.T)
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    im1 = axes[0].imshow(spe_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Spectral Feature Correlation Matrix')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(spa_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Spatial Feature Correlation Matrix')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_complementarity(spe_f, spa_f, save_path, num_features=30):
    """可视化特征互补性热图"""
    # 采样特征
    spe_sample = spe_f[:, :num_features].view(spe_f.size(0), -1).cpu().numpy()
    spa_sample = spa_f[:, :num_features].view(spa_f.size(0), -1).cpu().numpy()
    
    # 计算互补性指标（1 - 绝对相关系数）
    complementarity = 1 - np.abs(np.corrcoef(
        np.concatenate([spe_sample, spa_sample], axis=1).T
    )[:num_features, num_features:])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(complementarity, cmap='viridis', 
                xticklabels=[f'Spa_{i}' for i in range(num_features)],
                yticklabels=[f'Spe_{i}' for i in range(num_features)])
    plt.title('Feature Complementarity Heatmap (1 - |correlation|)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_distributions(spe_f, spa_f, save_path, num_channels=10):
    """对比频谱特征和空间特征的分布"""
    plt.figure(figsize=(15, 10))
    
    # 随机选择一些通道进行可视化
    spe_channels = np.random.choice(spe_f.shape[1], num_channels, replace=False)
    spa_channels = np.random.choice(spa_f.shape[1], num_channels, replace=False)
    
    # 绘制频谱特征分布
    plt.subplot(2, 1, 1)
    for i, channel in enumerate(spe_channels):
        channel_data = spe_f[:, channel].flatten().cpu().numpy()
        sns.kdeplot(channel_data, label=f'Ch {channel}', alpha=0.7)
    plt.title('Spectral Feature Distributions')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 绘制空间特征分布
    plt.subplot(2, 1, 2)
    for i, channel in enumerate(spa_channels):
        channel_data = spa_f[:, channel].flatten().cpu().numpy()
        sns.kdeplot(channel_data, label=f'Ch {channel}', alpha=0.7)
    plt.title('Spatial Feature Distributions')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()



def plot_corr_heatmap(spe_f, spa_f, save_path="corr.png"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    spe = spe_f.detach().cpu().view(-1, spe_f.shape[1])
    spa = spa_f.detach().cpu().view(-1, spa_f.shape[1])
    corr = np.corrcoef(spe.T, spa.T)[:spe_f.shape[1], spe_f.shape[1]:]

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Spectral vs Spatial Channel Correlation")
    plt.savefig(save_path)
    plt.close()
