import torch
import torch.nn as nn
import torch.optim as optim
from src.models.spectral_transformer import SpectralTemporalTransformer
from src.data_utils.dataset import SpectralDataset

def debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 只取 1 个样本！
    dataset = SpectralDataset("data/processed/spectral_data_v2.npz", max_nodes=20, max_seq_len=40)
    data = dataset[0] # 取第一个
    
    # 增加 Batch 维度
    evals = data['evals'].unsqueeze(0).to(device)
    evecs = data['evecs'].unsqueeze(0).to(device)
    time_idx = data['time_indices'].unsqueeze(0).to(device)
    targets = data['betas'].unsqueeze(0).to(device)
    mask = data['mask'].unsqueeze(0).to(device)
    
    # 2. 模型
    model = SpectralTemporalTransformer(max_nodes=20, d_model=128, max_seq_len=40).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # 较高的学习率
    criterion = nn.MSELoss()
    
    print("开始单样本过拟合测试 (Target: Loss -> 0)...")
    for epoch in range(500): # 跑 500 轮，死磕这一个数据
        optimizer.zero_grad()
        preds = model(evals, evecs, time_idx)
        loss = criterion(preds * mask, targets * mask)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.8f}")
            
    # 3. 验证
    print("\n测试结果:")
    print("真实值 (前5步):", targets[0, :5].cpu().detach().numpy())
    print("预测值 (前5步):", preds[0, :5].cpu().detach().numpy())

if __name__ == "__main__":
    debug()
