"""
噪声鲁棒性评估
分析神经网络预测在不同噪声条件下的表现
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.spectral_transformer import SpectralTemporalTransformer
from src.data_utils.dataset import SpectralDataset

# Ensure CJK font for Chinese tick labels
import matplotlib.font_manager as fm

def ensure_cjk_font():
    candidates = [
        'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK KR', 'SimHei',
        'WenQuanYi Micro Hei', 'Microsoft YaHei', 'PingFang SC', 'AR PL KaitiM GB'
    ]
    for f in fm.fontManager.ttflist:
        try:
            if any(name in f.name for name in candidates):
                plt.rcParams['font.sans-serif'] = [f.name]
                plt.rcParams['axes.unicode_minus'] = False
                return
        except Exception:
            continue
    local_font = os.path.join('assets', 'fonts', 'NotoSansSC-Regular.otf')
    if os.path.exists(local_font):
        try:
            fm.fontManager.addfont(local_font)
            name = fm.FontProperties(fname=local_font).get_name()
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Using local font: {name}")
            return
        except Exception as e:
            print(f"Failed to register local font: {e}")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("Warning: no CJK font found on system; Chinese labels may not render. Install 'fonts-noto-cjk' or similar.")

# apply font settings early
ensure_cjk_font()


def load_noisy_data(path, beta_mean, beta_std, max_nodes=20, max_seq_len=40):
    """加载噪声测试数据"""
    data = np.load(path, allow_pickle=True)['data']
    
    processed = []
    for item in data:
        evals = item['evals']
        evecs = item['evecs']
        # 兼容不同字段名
        if 'betas_clean' in item:
            betas_clean = item['betas_clean']
        elif 'betas' in item:
            betas_clean = item['betas']
        elif 'betas_raw' in item:
            betas_clean = item['betas_raw']
        else:
            betas_clean = np.array([], dtype=np.float32)
        
        if 'betas_noisy' in item:
            betas_noisy = item['betas_noisy']
        else:
            betas_noisy = betas_clean
        
        if 'energies_clean' in item:
            energies_clean = item['energies_clean']
        elif 'energies' in item:
            energies_clean = item['energies']
        else:
            energies_clean = np.array([], dtype=np.float32)
        
        energies_noisy = item.get('energies_noisy', energies_clean)
        N = evals.shape[0]
        
        # Padding
        evals_pad = np.zeros(max_nodes, dtype=np.float32)
        evals_pad[:N] = evals
        
        evecs_pad = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        evecs_pad[:N, :N] = evecs
        
        processed.append({
            'evals': torch.from_numpy(evals_pad),
            'evecs': torch.from_numpy(evecs_pad),
            'betas_clean': betas_clean,
            'betas_noisy': betas_noisy,
            'energies_clean': energies_clean,
            'energies_noisy': energies_noisy,
            'num_nodes': N,
            'graph_type': item.get('graph_type', 'unknown'),
            'noise_config': item.get('noise_config', {})
        })
    
    return processed


def evaluate_with_noise(model, data, device, beta_mean, beta_std):
    """评估模型在噪声数据上的表现"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for item in data:
            evals = item['evals'].unsqueeze(0).to(device)
            evecs = item['evecs'].unsqueeze(0).to(device)
            time_idx = torch.arange(40).unsqueeze(0).to(device)
            num_nodes = torch.tensor([item['num_nodes']]).to(device)
            
            # 模型预测
            pred = model.generate(evals, evecs, time_idx, num_nodes=num_nodes)
            pred = pred.cpu().numpy()[0]
            pred = pred * beta_std + beta_mean
            
            betas_clean = item['betas_clean']
            betas_noisy = item['betas_noisy']
            
            valid_len = min(len(betas_clean), 40)
            pred_slice = pred[:valid_len]
            clean_slice = betas_clean[:valid_len]
            noisy_slice = betas_noisy[:valid_len]
            
            # 计算指标
            # 1. 模型预测 vs 干净标签
            if np.std(clean_slice) > 1e-6 and np.std(pred_slice) > 1e-6:
                corr_pred_clean = np.corrcoef(pred_slice, clean_slice)[0, 1]
            else:
                corr_pred_clean = np.nan
            
            # 2. 噪声 FALQON vs 干净标签
            if np.std(clean_slice) > 1e-6 and np.std(noisy_slice) > 1e-6:
                corr_noisy_clean = np.corrcoef(noisy_slice, clean_slice)[0, 1]
            else:
                corr_noisy_clean = np.nan
            
            # 3. 能量比较
            final_energy_pred = item.get('energies_clean')[-1] if item.get('energies_clean') is not None else np.nan
            final_energy_noisy = item.get('energies_noisy')[-1] if item.get('energies_noisy') is not None else np.nan
            
            results.append({
                'num_nodes': item['num_nodes'],
                'corr_pred_clean': corr_pred_clean,
                'corr_noisy_clean': corr_noisy_clean,
                'mae_pred': np.mean(np.abs(pred_slice - clean_slice)),
                'mae_noisy': np.mean(np.abs(noisy_slice - clean_slice)),
                'energy_clean': final_energy_pred,
                'energy_noisy': final_energy_noisy,
            })
    
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载训练集统计量
    train_dataset = SpectralDataset("data/processed/spectral_data_v2.npz")
    beta_mean = train_dataset.beta_mean
    beta_std = train_dataset.beta_std
    
    # 加载模型
    model = SpectralTemporalTransformer(
        max_nodes=20, d_model=128, nhead=4, num_layers=4, max_seq_len=40
    ).to(device)
    try:
        model.load_state_dict(torch.load("models/spectral_transformer_finetuned.pth", map_location=device))
    except RuntimeError as e:
        print(f"Warning: model.load_state_dict failed: {e}. Retrying with strict=False.")
        model.load_state_dict(torch.load("models/spectral_transformer_finetuned.pth", map_location=device), strict=False)
    
    # 噪声级别
    noise_levels = [
        ("no_noise", "无噪声"),
        ("low_noise", "低噪声"),
        ("medium_noise", "中等噪声"),
        ("high_noise", "高噪声"),
        ("extreme_noise", "极端噪声"),
    ]
    
    summary = []
    
    for noise_name, noise_label in noise_levels:
        path = f"data/noise_test/{noise_name}.npz"
        if not os.path.exists(path):
            print(f"跳过 {noise_label}: 文件不存在")
            continue
        
        print(f"\n评估 {noise_label}...")
        data = load_noisy_data(path, beta_mean, beta_std)
        results = evaluate_with_noise(model, data, device, beta_mean, beta_std)
        
        # 统计
        corr_pred = np.nanmean([r['corr_pred_clean'] for r in results]) if results else float('nan')
        corr_noisy = np.nanmean([r['corr_noisy_clean'] for r in results]) if results else float('nan')
        mae_pred = np.mean([r['mae_pred'] for r in results]) if results else float('nan')
        mae_noisy = np.mean([r['mae_noisy'] for r in results]) if results else float('nan')
        
        print(f"  神经网络预测 vs 干净标签: Corr={corr_pred:.3f}, MAE={mae_pred:.3f}")
        print(f"  噪声FALQON vs 干净标签: Corr={corr_noisy:.3f}, MAE={mae_noisy:.3f}")
        print(f"  → 神经网络优势: ΔCorr={corr_pred - corr_noisy:+.3f}")
        
        summary.append({
            'noise_level': noise_label,
            'corr_nn': corr_pred,
            'corr_noisy_falqon': corr_noisy,
            'mae_nn': mae_pred,
            'mae_noisy_falqon': mae_noisy,
        })
    
    # 绘图
    os.makedirs("output/noise", exist_ok=True)
    
    if summary:
        # 图1: 相关系数对比
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(summary))
        width = 0.35
        
        labels = [s['noise_level'] for s in summary]
        corr_nn = [s['corr_nn'] for s in summary]
        corr_noisy = [s['corr_noisy_falqon'] for s in summary]
        
        bars1 = ax.bar(x - width/2, corr_nn, width, label='Neural Network (Ours)', color='steelblue')
        bars2 = ax.bar(x + width/2, corr_noisy, width, label='Noisy FALQON', color='coral')
        
        ax.set_ylabel('Correlation with Clean Reference', fontsize=12)
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_title('Noise Robustness: Neural Network vs Hardware FALQON', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars1, corr_nn):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, corr_noisy):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('output/noise/noise_robustness_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 图2: 优势曲线
        fig, ax = plt.subplots(figsize=(8, 5))
        
        advantage = [s['corr_nn'] - s['corr_noisy_falqon'] for s in summary]
        ax.plot(labels, advantage, 'o-', color='green', linewidth=2, markersize=10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(labels, 0, advantage, alpha=0.3, color='green')
        
        ax.set_ylabel('Correlation Advantage (NN - Noisy FALQON)', fontsize=12)
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_title('Neural Network Advantage Under Different Noise Conditions', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/noise/nn_advantage_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n图片已保存到 output/noise/")

if __name__ == "__main__":
    main()
