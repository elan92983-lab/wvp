"""
跨规模泛化评估
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.spectral_transformer import SpectralTemporalTransformer
from src.data_utils.dataset import SpectralDataset


def load_test_data(path, beta_mean, beta_std, max_nodes=20, max_seq_len=40):
    """加载并预处理测试数据"""
    data = np.load(path, allow_pickle=True)['data']
    
    processed = []
    for item in data:
        evals = item['evals']
        evecs = item['evecs']
        # 兼容不同字段名
        if 'betas' in item:
            betas = item['betas']
        elif 'betas_clean' in item:
            betas = item['betas_clean']
        elif 'betas_raw' in item:
            betas = item['betas_raw']
        else:
            betas = np.array([], dtype=np.float32)
        N = evals.shape[0]
        
        if N > max_nodes:
            # 截断到 max_nodes
            evals = evals[:max_nodes]
            evecs = evecs[:max_nodes, :max_nodes]
        
        # Padding
        evals_pad = np.zeros(max_nodes, dtype=np.float32)
        evals_pad[:len(evals)] = evals
        
        evecs_pad = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        evecs_pad[:evecs.shape[0], :evecs.shape[1]] = evecs
        
        betas_norm = (betas - beta_mean) / beta_std
        betas_pad = np.zeros(max_seq_len, dtype=np.float32)
        real_len = min(len(betas), max_seq_len)
        betas_pad[:real_len] = betas_norm[:real_len]
        
        processed.append({
            'evals': torch.from_numpy(evals_pad),
            'evecs': torch.from_numpy(evecs_pad),
            'betas_raw': betas,
            'num_nodes': N,
            'graph_type': item.get('graph_type', 'unknown'),
        })
    
    return processed


def evaluate_model(model, data, device, beta_mean, beta_std):
    """评估模型"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for item in data:
            evals = item['evals'].unsqueeze(0).to(device)
            evecs = item['evecs'].unsqueeze(0).to(device)
            time_idx = torch.arange(40).unsqueeze(0).to(device)
            num_nodes = torch.tensor([min(item['num_nodes'], 20)]).to(device)
            
            pred = model.generate(evals, evecs, time_idx, num_nodes=num_nodes)
            pred = pred.cpu().numpy()[0]
            pred = pred * beta_std + beta_mean
            
            real = item['betas_raw']
            valid_len = min(len(real), 40)
            
            pred_slice = pred[:valid_len]
            real_slice = real[:valid_len]
            
            if np.std(real_slice) > 1e-6 and np.std(pred_slice) > 1e-6:
                corr = np.corrcoef(pred_slice, real_slice)[0, 1]
            else:
                corr = np.nan
            
            mae = np.mean(np.abs(pred_slice - real_slice))
            
            results.append({
                'num_nodes': item['num_nodes'],
                'graph_type': item['graph_type'],
                'corr': corr,
                'mae': mae,
                'pred': pred_slice,
                'real': real_slice
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
    
    # Test datasets (English labels)
    test_sets = [
        ("In-domain (6-13)", "data/scalability_test/in_domain.npz"),
        ("Mild extrapolation (14-17)", "data/scalability_test/mild_extrap.npz"),
        ("Strong extrapolation (18-22)", "data/scalability_test/strong_extrap.npz"),
        ("Extreme extrapolation (23-28)", "data/scalability_test/extreme_extrap.npz"),
    ]
    
    all_results = {}
    
    for name, path in test_sets:
        if not os.path.exists(path):
            print(f"跳过 {name}: 文件不存在")
            continue
        
        print(f"\n评估 {name}...")
        data = load_test_data(path, beta_mean, beta_std)
        results = evaluate_model(model, data, device, beta_mean, beta_std)
        all_results[name] = results
        
        # 按图类型统计
        by_type = defaultdict(list)
        for r in results:
            by_type[r['graph_type']].append(r)
        
        print(f"  总样本数: {len(results)}")
        for gtype, gresults in by_type.items():
            corrs = [r['corr'] for r in gresults if not np.isnan(r['corr'])]
            print(f"  {gtype}: n={len(gresults)}, Corr={np.nanmean(corrs) if corrs else float('nan'):.3f}±{np.nanstd(corrs) if corrs else float('nan'):.3f}")
    
    # 绘图
    os.makedirs("output/scalability", exist_ok=True)
    
    # 图1: Corr vs N 散点图
    plt.figure(figsize=(12, 6))
    colors = {'er': 'blue', 'regular': 'red', 'unknown': 'gray', 'mixed': 'green'}
    
    for name, results in all_results.items():
        for r in results:
            color = colors.get(r['graph_type'], 'gray')
            plt.scatter(r['num_nodes'], r['corr'], c=color, alpha=0.5, s=30)
    
    plt.xlabel('Number of Nodes (N)', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.title('Cross-Scale Generalization: Prediction Accuracy vs Graph Size', fontsize=14)
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.8)')
    plt.legend(['ER Graph', '3-Regular Graph', 'Threshold'])
    plt.grid(True, alpha=0.3)
    plt.savefig('output/scalability/corr_vs_n.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图2: 按 N 范围的箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    box_data = []
    labels = []
    for name, results in all_results.items():
        corrs = [r['corr'] for r in results if not np.isnan(r['corr'])]
        box_data.append(corrs)
        labels.append(name.split('(')[0].strip())
    
    if any(len(b) > 0 for b in box_data):
        ax.boxplot(box_data, labels=labels)
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_title('Generalization Performance Across Different Scale Ranges', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        plt.savefig('output/scalability/boxplot_by_range.png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print('没有足够数据生成箱线图')
    
    print("\n图片已保存到 output/scalability/")

if __name__ == "__main__":
    main()
