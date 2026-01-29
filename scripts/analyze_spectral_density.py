"""
谱密度分析：研究谱分布相似性与模型泛化能力的关系
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance
from scipy.special import ellipk
import os

def compute_spectral_histogram(evals_list, bins=50, range_limit=(0, 2)):
    """计算特征值直方图"""
    all_evals = np.concatenate([e.flatten() for e in evals_list])
    hist, bin_edges = np.histogram(all_evals, bins=bins, range=range_limit, density=True)
    return hist, bin_edges

def kl_divergence(p, q, epsilon=1e-10):
    """KL散度"""
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / p.sum()
    q = q / q.sum()
    return entropy(p, q)

def wigner_semicircle(x, R=1.0):
    """Wigner 半圆分布（ER图的渐近谱密度）"""
    result = np.zeros_like(x)
    mask = np.abs(x - 1) < R
    result[mask] = (2 / (np.pi * R**2)) * np.sqrt(R**2 - (x[mask] - 1)**2)
    return result

def kesten_mckay(x, d=3):
    """Kesten-McKay 分布（d-正则图的渐近谱密度）"""
    # 对于归一化拉普拉斯，特征值在 [0, 2] 范围
    # 这里使用简化的近似
    result = np.zeros_like(x)
    center = 1.0
    width = 2 * np.sqrt(d - 1) / d
    mask = np.abs(x - center) < width
    if mask.sum() > 0:
        result[mask] = (d / (2 * np.pi)) * np.sqrt(4 * (d - 1) / d**2 - (x[mask] - center)**2)
        result[mask] /= (result[mask].sum() * (x[1] - x[0]) + 1e-10)  # 归一化
    return result

def main():
    output_dir = "output/spectral_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载训练数据
    train_path = "data/processed/spectral_data_v2.npz"
    if not os.path.exists(train_path):
        print(f"跳过: 训练数据 {train_path} 不存在")
        return
    train_data = np.load(train_path, allow_pickle=True)['data']
    
    # 按节点数和图类型分组
    data_by_n = {}
    for item in train_data:
        n = item['evals'].shape[0]
        if n not in data_by_n:
            data_by_n[n] = {'er': [], 'regular': [], 'all': []}
        
        data_by_n[n]['all'].append(item['evals'])
        # 注意：原始数据可能没有 graph_type 字段
    
    # 图1: 不同 N 的谱密度演化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    n_values = sorted(data_by_n.keys())[:8]
    bins = 40
    x = np.linspace(0, 2, bins)
    
    for idx, n in enumerate(n_values):
        evals_list = data_by_n[n]['all']
        hist, bin_edges = compute_spectral_histogram(evals_list, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        axes[idx].bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], 
                      alpha=0.7, label='Empirical')
        
        # 理论曲线
        semicircle = wigner_semicircle(bin_centers)
        axes[idx].plot(bin_centers, semicircle / (semicircle.sum() * (bin_centers[1] - bin_centers[0]) + 1e-10) * hist.sum() * (bin_centers[1] - bin_centers[0]), 
                       'r--', linewidth=2, label='Wigner Semicircle')
        
        axes[idx].set_title(f'N = {n} ({len(evals_list)} samples)', fontsize=11)
        axes[idx].set_xlabel('λ')
        axes[idx].set_ylabel('Density')
        axes[idx].set_xlim(0, 2)
        if idx == 0:
            axes[idx].legend(fontsize=8)
    
    plt.suptitle('Spectral Density Evolution with Graph Size', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spectral_density_by_n.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图2: KL散度矩阵
    kl_matrix = np.zeros((len(n_values), len(n_values)))
    
    for i, n1 in enumerate(n_values):
        hist1, _ = compute_spectral_histogram(data_by_n[n1]['all'], bins=bins)
        for j, n2 in enumerate(n_values):
            hist2, _ = compute_spectral_histogram(data_by_n[n2]['all'], bins=bins)
            kl_matrix[i, j] = kl_divergence(hist1, hist2)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(kl_matrix, cmap='Blues')
    plt.colorbar(im, label='KL Divergence')
    ax.set_xticks(range(len(n_values)))
    ax.set_yticks(range(len(n_values)))
    ax.set_xticklabels(n_values)
    ax.set_yticklabels(n_values)
    ax.set_xlabel('Graph Size N')
    ax.set_ylabel('Graph Size N')
    ax.set_title('Spectral Density KL Divergence Matrix', fontsize=14)
    
    for i in range(len(n_values)):
        for j in range(len(n_values)):
            ax.text(j, i, f'{kl_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kl_divergence_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图3: 与理论极限的收敛
    fig, ax = plt.subplots(figsize=(10, 6))
    
    distances_to_semicircle = []
    for n in n_values:
        hist, bin_edges = compute_spectral_histogram(data_by_n[n]['all'], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        theoretical = wigner_semicircle(bin_centers)
        theoretical = theoretical / (theoretical.sum() + 1e-10)
        hist = hist / (hist.sum() + 1e-10)
        
        # Wasserstein 距离
        w_dist = wasserstein_distance(bin_centers, bin_centers, hist, theoretical)
        distances_to_semicircle.append(w_dist)
    
    ax.plot(n_values, distances_to_semicircle, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Graph Size N', fontsize=12)
    ax.set_ylabel('Wasserstein Distance to Semicircle Law', fontsize=12)
    ax.set_title('Convergence of Spectral Density to Theoretical Limit', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence_to_semicircle.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存到 {output_dir}/")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("谱密度分析结果")
    print("="*60)
    print(f"\n不同 N 之间的 KL 散度范围: {kl_matrix.min():.4f} - {kl_matrix.max():.4f}")
    print(f"对角线外最大 KL 散度: {np.max(kl_matrix[~np.eye(len(n_values), dtype=bool)]):.4f}")
    print(f"\n与半圆律的 Wasserstein 距离:")
    for n, d in zip(n_values, distances_to_semicircle):
        print(f"  N={n}: {d:.4f}")

if __name__ == "__main__":
    main()
