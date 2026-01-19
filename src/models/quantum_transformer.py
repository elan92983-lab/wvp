import torch
import torch.nn as nn

class GadgetTokenizedTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_layers, seq_len):
        super().__init__()
        # 创新点一：超级符令嵌入层
        # 这里将图的邻接矩阵特征映射到嵌入空间
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # 核心：Transformer 解码器结构，用于序列预测
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出层：预测每一层的 Beta 值
        self.fc_out = nn.Linear(model_dim, 1) 
        self.seq_len = seq_len

    def forward(self, graph_features):
        # graph_features 形状: (batch, input_dim)
        embed = self.embedding(graph_features).unsqueeze(0) # (1, batch, model_dim)
        
        # 生成预测序列
        # 这里的逻辑对应于“教师-学生”模型中的参数预测
        output = self.transformer_decoder(embed, embed) # 简化版自回归
        return self.fc_out(output).squeeze()
