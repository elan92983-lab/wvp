import torch
import torch.nn as nn
import math

class GraphEncoder(nn.Module):
    """
    简单的图编码器：将邻接矩阵的每一行视为一个节点的初始特征
    """
    def __init__(self, max_nodes, d_model):
        super().__init__()
        # 输入维度是 max_nodes (因为邻接矩阵的一行有 max_nodes 个元素)
        self.projection = nn.Linear(max_nodes, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, adj):
        # adj: [Batch, N, N]
        # x: [Batch, N, d_model]
        x = self.projection(adj)
        x = self.norm(x)
        return x

class FALQONTransformer(nn.Module):
    def __init__(self, 
                 max_nodes=12, 
                 output_len=30, 
                 d_model=128, 
                 nhead=4, 
                 num_layers=3, 
                 dim_feedforward=512, 
                 dropout=0.1):
        super().__init__()
        
        self.model_type = 'Transformer'
        
        # 1. 输入嵌入层
        self.encoder_embedding = GraphEncoder(max_nodes, d_model)
        
        # 2. 位置编码 (对于图节点来说，位置编码可能不是必须的，但在 Permutation Invariant 还没完全保证时加上无妨)
        # 这里我们使用一个可学习的位置 Embedding，以此区分不同的节点槽位
        self.pos_encoder = nn.Parameter(torch.randn(1, max_nodes, d_model) * 0.1)

        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 4. 输出头 (Decoder)
        # 策略：将 Transformer 输出的所有节点特征取平均 (Mean Pooling)，得到图的全局特征
        # 然后通过 MLP 映射到 30 个 beta 值
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_len)  # 输出 30 个 beta
        )

    def forward(self, src, src_key_padding_mask=None):
        """
        src: [Batch, N, N] (邻接矩阵)
        src_key_padding_mask: [Batch, N] (0 表示填充节点，需要被忽略)
        """
        # [Batch, N, d_model]
        x = self.encoder_embedding(src)
        
        # 加上位置编码
        x = x + self.pos_encoder
        
        # Transformer 编码
        # 注意：PyTorch 的 mask 逻辑是 True 代表"被忽略/填充"，False 代表"保留"
        # 我们的 dataset 里 mask 是 1(真实), 0(填充)。所以要取反。
        if src_key_padding_mask is not None:
            # 转换 mask: 1->False (保留), 0->True (忽略)
            pytorch_mask = (src_key_padding_mask == 0)
        else:
            pytorch_mask = None

        # x shape: [Batch, N, d_model]
        memory = self.transformer_encoder(x, src_key_padding_mask=pytorch_mask)

        # Global Pooling: 对所有有效节点的特征取平均
        # 简单起见，我们对第 0 维(Batch)外的所有节点取平均，但要注意填充节点不能算进去
        # 这里为了演示简单，直接用 Mask 加权平均
        
        if src_key_padding_mask is not None:
            # [Batch, N, 1]
            mask_expanded = src_key_padding_mask.unsqueeze(-1)
            # 求和时只算真实节点
            sum_embeddings = torch.sum(memory * mask_expanded, dim=1) 
            # 真实节点数量
            num_valid = torch.sum(mask_expanded, dim=1).clamp(min=1)
            global_feature = sum_embeddings / num_valid
        else:
            global_feature = torch.mean(memory, dim=1)

        # 预测 Betas
        # [Batch, 30]
        output = self.output_head(global_feature)
        
        return output

# 测试代码
if __name__ == "__main__":
    model = FALQONTransformer()
    # 模拟一个 Batch=2, 节点数=12 的输入
    dummy_adj = torch.randn(2, 12, 12)
    dummy_mask = torch.ones(2, 12) 
    dummy_mask[0, 10:] = 0 # 第一个样本只有 10 个节点有效
    
    out = model(dummy_adj, dummy_mask)
    print("模型输出形状:", out.shape) # 应该是 [2, 30]
