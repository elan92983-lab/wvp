import torch
import torch.nn as nn
import math

class SignNet(nn.Module):
    """
    1.26.md 7.1 节提到的 SignNet。
    用于处理特征向量的符号模糊性: f(v) = f(-v)。
    实现方式: Phi(v) + Phi(-v)
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # 最后的聚合层
        self.rho = nn.Linear(out_dim, out_dim)

    def forward(self, evecs):
        # evecs shape: [Batch, N, N] -> 视为 [Batch, N, feature_dim=N]
        # 我们对每一行（每个节点的特征向量分量）进行编码
        
        # 这里的 SignNet 变体：我们希望对整个特征向量矩阵具备符号不变性
        # 简化版：对每个特征向量 v_i (列) 进行处理
        # 实际上，DeepSets 思想: Phi(v) + Phi(-v)
        
        # 维度变换: [B, N, N] -> [B*N, N]
        B, N, _ = evecs.shape
        flat_evecs = evecs.view(B * N, N)
        
        # Phi(v) + Phi(-v)
        h1 = self.phi(flat_evecs)
        h2 = self.phi(-flat_evecs)
        h = h1 + h2 
        
        h = self.rho(h)
        # 还原维度 [B, N, dim]
        return h.view(B, N, -1)

class SinusoidalPositionalEncoding(nn.Module):
    """
    1.26.md 提到的时序位置编码，作为 Query 输入。
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, time_steps):
        # time_steps: [Batch, Seq_Len] (索引)
        # Output: [Batch, Seq_Len, d_model]
        return self.pe[time_steps]

class SpectralTemporalTransformer(nn.Module):
    """
    对应 1.26.md 报告中的核心架构。
    """
    def __init__(self, max_nodes=20, d_model=128, nhead=4, num_layers=3, max_seq_len=40):
        super().__init__()
        self.d_model = d_model
        
        # --- 1. 谱编码模块 (Spectral Encoder) ---
        # 特征值编码
        self.eval_encoder = nn.Linear(max_nodes, d_model)
        # 特征向量编码 (SignNet)
        self.sign_net = SignNet(max_nodes, d_model, d_model)
        # 融合层
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # --- 2. 时序解码模块 (Temporal Decoder) ---
        self.time_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len + 10)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # --- 3. 输出头 ---
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 输出标量 beta
        )

    def forward(self, evals, evecs, time_indices):
        """
        Args:
            evals: [Batch, N] 特征值 (需补零对齐到 max_nodes)
            evecs: [Batch, N, N] 特征向量
            time_indices: [Batch, P] 时间步索引 (0, 1, ..., P-1)
        """
        # --- 1. 构建 Memory (图谱信息) ---
        # 补零处理已经在 Dataset 中做完，假设输入固定维度
        
        # 编码特征值
        h_eval = self.eval_encoder(evals) # [B, d_model]
        h_eval = h_eval.unsqueeze(1).expand(-1, evecs.shape[1], -1) # [B, N, d_model]
        
        # 编码特征向量
        h_evec = self.sign_net(evecs) # [B, N, d_model]
        
        # 融合: 这里的 Memory 长度是 N (节点数)，代表谱空间的模态
        memory = self.fusion(torch.cat([h_eval, h_evec], dim=-1)) # [B, N, d_model]
        
        # --- 2. 构建 Query (时间信息) ---
        # Target 是时间序列
        tgt = self.time_encoding(time_indices) # [B, P, d_model]
        
        # --- 3. Transformer 解码 ---
        # Cross-Attention: Query=Time, Key/Value=Spectral Memory
        out = self.transformer_decoder(tgt, memory) # [B, P, d_model]
        
        # --- 4. 预测参数 ---
        beta_pred = self.head(out).squeeze(-1) # [B, P]
        
        return beta_pred
