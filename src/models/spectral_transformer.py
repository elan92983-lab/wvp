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
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # 最后的聚合层
        self.rho = nn.Linear(out_dim, out_dim)

    def forward(self, evecs):
        # evecs shape: [Batch, M, N] (M 为选择的特征向量个数, N 为节点数)
        # 这里对每个特征向量 v 进行编码，保证 f(v) = f(-v)

        # 维度变换: [B, M, N] -> [B*M, N]
        B, M, N = evecs.shape
        flat_evecs = evecs.reshape(B * M, N)
        
        # Phi(v) + Phi(-v)
        h1 = self.phi(flat_evecs)
        h2 = self.phi(-flat_evecs)
        h = h1 + h2 
        
        h = self.rho(h)
        # 还原维度 [B, N, dim]
        return h.view(B, M, -1)

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
    def __init__(self, max_nodes=20, d_model=128, nhead=4, num_layers=3, max_seq_len=40, num_modes=None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.max_nodes = max_nodes
        self.num_modes = num_modes or max_nodes
        # --- 1. 谱编码模块 (Spectral Encoder) ---
        self.eval_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.sign_net = SignNet(max_nodes, d_model, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        # --- 2. 时序解码模块 (Temporal Decoder) ---
        self.time_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len + 10)
        # === 新增：可学习的 Query 嵌入 ===
        self.query_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        # === 新增：图全局特征的专属 Token ===
        self.graph_token_embed = nn.Parameter(torch.randn(1, 1, d_model))
        # === 新增：前一步 beta 嵌入（自回归） ===
        self.beta_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # --- 3. 输出头 ---
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, evals, evecs, time_indices, num_nodes=None, prev_betas=None):
        B = evals.shape[0]
        # --- 1. 构建 Memory (谱模态信息) ---
        num_modes = min(self.num_modes, evals.shape[1])
        evals = evals[:, :num_modes]
        evecs = evecs[:, :, :num_modes]

        # evals: [B, M] -> [B, M, 1] -> [B, M, d_model]
        h_eval = self.eval_encoder(evals.unsqueeze(-1))
        # evecs: [B, N, M] -> [B, M, N]
        h_evec = self.sign_net(evecs.transpose(1, 2))
        
        # --- 1.5 将图的全局信息作为一个特殊的 Token 加入 Memory ---
        if num_nodes is not None:
            valid_modes = torch.clamp(num_nodes, max=num_modes)
            mode_ids = torch.arange(num_modes, device=evals.device).unsqueeze(0)
            mode_mask = (mode_ids < valid_modes.unsqueeze(1)).unsqueeze(-1)
            # 基于有效 mode 计算 memory
            mode_memory = self.fusion(torch.cat([h_eval, h_evec], dim=-1))
            masked_sum = (mode_memory * mode_mask).sum(dim=1)
            denom = mode_mask.sum(dim=1).clamp_min(1.0)
            graph_global = masked_sum / denom
        else:
            mode_memory = self.fusion(torch.cat([h_eval, h_evec], dim=-1))
            graph_global = mode_memory.mean(dim=1)

        # 将全局图特征与可学习的 token 嵌入结合，并扩展到 batch size
        graph_token = self.graph_token_embed.expand(B, -1, -1) + graph_global.unsqueeze(1)
        
        # 最终的 memory 是 mode memory 和 graph token 的拼接
        memory = torch.cat([graph_token, mode_memory], dim=1)

        # 处理 padding：基于 num_nodes 生成 memory_key_padding_mask
        # 注意：graph_token 永远不被 mask
        memory_key_padding_mask = None
        if num_nodes is not None:
            num_nodes = num_nodes.to(evals.device)
            valid_modes = torch.clamp(num_nodes, max=num_modes)
            mode_ids = torch.arange(num_modes, device=evals.device).unsqueeze(0)
            mode_mask = mode_ids >= valid_modes.unsqueeze(1)
            # graph_token 对应的 mask 是 False
            graph_token_mask = torch.zeros((B, 1), dtype=torch.bool, device=evals.device)
            memory_key_padding_mask = torch.cat([graph_token_mask, mode_mask], dim=1)

        # --- 2. 构建 Query (只包含时序信息) ---
        time_pe = self.time_encoding(time_indices)  # [B, P, d_model]

        beta_pe = 0
        if prev_betas is not None:
            beta_pe = self.beta_embed(prev_betas.unsqueeze(-1))  # [B, P, d_model]

        tgt = self.query_embed.expand(B, -1, -1) + time_pe + beta_pe

        # --- 3. Transformer 解码 ---
        P = time_indices.shape[1]
        tgt_mask = torch.triu(
            torch.full((P, P), float('-inf'), device=evals.device),
            diagonal=1
        )
        out = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # [B, P, d_model]
        # --- 4. 预测参数 ---
        beta_delta = self.head(out).squeeze(-1)  # [B, P]
        if prev_betas is not None:
            beta_pred = beta_delta + prev_betas
        else:
            beta_pred = beta_delta
        return beta_pred

    @torch.no_grad()
    def generate(self, evals, evecs, time_indices, num_nodes=None):
        """自回归生成 beta 序列（推理用）"""
        B, P = time_indices.shape
        device = time_indices.device
        preds = torch.zeros((B, P), device=device)
        for t in range(P):
            # === CRITICAL FIX: Shift preds right to align with training logic ===
            # Training: prev_betas[:, t] = real_betas[:, t-1]
            # Inference: We must shift our current accumulated preds so that
            #            input index t contains the prediction from t-1.
            prev = torch.zeros_like(preds)
            prev[:, 1:] = preds[:, :-1]
            
            out = self.forward(evals, evecs, time_indices, num_nodes=num_nodes, prev_betas=prev)
            preds[:, t] = out[:, t]
        return preds
