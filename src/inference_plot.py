import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from src.models.transformer import FALQONTransformer
from src.algorithms.falqon_core import FALQON

# ⚠️ 服务器通常没有屏幕，必须设置这个后端才能保存图片
plt.switch_backend('Agg')

def main():
    # 1. 配置
    model_path = "models/checkpoints/best_model.pth"
    save_img_path = "output/prediction_result.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📦 加载模型: {model_path}")
    
    # 2. 初始化模型并加载权重
    # 注意：参数必须与训练时一致 (max_nodes=12, output_len=30)
    model = FALQONTransformer(max_nodes=12, output_len=30,d_model=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换到评估模式
    
    print("🎲 生成测试样本 (8个节点的随机图)...")
    # 生成一个新的随机图
    while True:
        num_nodes = 8
        g = nx.erdos_renyi_graph(num_nodes, p=0.5)
        if nx.is_connected(g):
            break
            
    # --- A. 获取标准答案 (Ground Truth) ---
    print("🐢 正在运行经典 FALQON (Teacher)...")
    teacher = FALQON(g, alpha=0.5)
    true_betas, _ = teacher.train(max_layers=30)
    
    # --- B. 获取 AI 预测 (Student) ---
    print("⚡ 正在运行 AI 预测 (Student)...")
    
    # 预处理图数据以符合模型输入
    adj = nx.to_numpy_array(g)
    padded_adj = np.zeros((12, 12), dtype=np.float32)
    padded_adj[:num_nodes, :num_nodes] = adj
    
    mask = np.zeros(12, dtype=np.float32)
    mask[:num_nodes] = 1.0
    
    # 转为 Tensor 并增加 Batch 维度
    adj_tensor = torch.tensor(padded_adj).unsqueeze(0).to(device).float() # [1, 12, 12]
    mask_tensor = torch.tensor(mask).unsqueeze(0).to(device).float()      # [1, 12]
    
    with torch.no_grad():
        pred_betas = model(adj_tensor, mask_tensor) # [1, 30]
        pred_betas = pred_betas.cpu().numpy().flatten()

    # --- C. 画图对比 ---
    print(f"🎨 正在绘图保存至 {save_img_path} ...")
    plt.figure(figsize=(10, 6))
    
    # 画线
    plt.plot(true_betas, 'b-o', label='Ground Truth (FALQON)', linewidth=2, alpha=0.7)
    plt.plot(pred_betas, 'r--x', label='AI Prediction (Transformer)', linewidth=2, alpha=0.9)
    
    plt.title(f"FALQON Parameter Prediction (N={num_nodes})", fontsize=14)
    plt.xlabel("Layer Step", fontsize=12)
    plt.ylabel("Beta Value", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存
    os.makedirs("output", exist_ok=True)
    plt.savefig(save_img_path, dpi=300)
    print("✅ 完成！")

if __name__ == "__main__":
    main()
