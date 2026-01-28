import torch
import torch.nn as nn
import math

class SignNet(nn.Module):
    """
    处理特征向量的符号模糊性: f(v) = f(-v)
    增强版：更深的网络
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.rho = nn.Linear(out_dim, out_dim)

    def forward(self, evecs):
        B, M, N = evecs.shape
        flat_evecs = evecs.reshape(B * M, N)
        h1 = self.phi(flat_evecs)
        h2 = self.phi(-flat_evecs)
        h = h1 + h2
        h = self.rho(h)
        return h.view(B, M, -1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, time_steps):
        return self.pe[time_steps]


class SpectralTemporalTransformerV2(nn.Module):
    """
    增强版 Spectral-Temporal Transformer
    """
    def __init__(self, max_nodes=20, d_model=256, nhead=8, num_layers=6,
                 max_seq_len=40, num_modes=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.max_nodes = max_nodes
        self.num_modes = num_modes or max_nodes

        # --- 1. 谱编码模块 (增强版) ---
        self.eval_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.sign_net = SignNet(max_nodes, d_model, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # --- 2. 时序解码模块 ---
        self.time_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len + 10)
        self.query_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.graph_token_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.beta_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- 输出头 ---
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))

    def forward(self, evals, evecs, time_indices, num_nodes=None, prev_betas=None):
        B = evals.shape[0]
        num_modes = min(self.num_modes, evals.shape[1])
        evals = evals[:, :num_modes]
        evecs = evecs[:, :, :num_modes]

        h_eval = self.eval_encoder(evals.unsqueeze(-1))
        h_evec = self.sign_net(evecs.transpose(1, 2))

        if num_nodes is not None:
            valid_modes = torch.clamp(num_nodes, max=num_modes)
            mode_ids = torch.arange(num_modes, device=evals.device).unsqueeze(0)
            mode_mask = (mode_ids < valid_modes.unsqueeze(1)).unsqueeze(-1)
            mode_memory = self.fusion(torch.cat([h_eval, h_evec], dim=-1))
            masked_sum = (mode_memory * mode_mask).sum(dim=1)
            denom = mode_mask.sum(dim=1).clamp_min(1.0)
            graph_global = masked_sum / denom
        else:
            mode_memory = self.fusion(torch.cat([h_eval, h_evec], dim=-1))
            graph_global = mode_memory.mean(dim=1)

        graph_token = self.graph_token_embed.expand(B, -1, -1) + graph_global.unsqueeze(1)
        memory = torch.cat([graph_token, mode_memory], dim=1)

        memory_key_padding_mask = None
        if num_nodes is not None:
            valid_modes = torch.clamp(num_nodes.to(evals.device), max=num_modes)
            mode_ids = torch.arange(num_modes, device=evals.device).unsqueeze(0)
            mode_mask = mode_ids >= valid_modes.unsqueeze(1)
            graph_token_mask = torch.zeros((B, 1), dtype=torch.bool, device=evals.device)
            memory_key_padding_mask = torch.cat([graph_token_mask, mode_mask], dim=1)

        time_pe = self.time_encoding(time_indices)
        beta_pe = 0
        if prev_betas is not None:
            beta_pe = self.beta_embed(prev_betas.unsqueeze(-1))

        tgt = self.query_embed.expand(B, -1, -1) + time_pe + beta_pe

        P = time_indices.shape[1]
        tgt_mask = torch.triu(
            torch.full((P, P), float('-inf'), device=evals.device),
            diagonal=1
        )
        out = self.transformer_decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        beta_delta = self.head(out).squeeze(-1) * self.output_scale + self.output_bias
        if prev_betas is not None:
            beta_pred = beta_delta + prev_betas
        else:
            beta_pred = beta_delta
        return beta_pred

    @torch.no_grad()
    def generate(self, evals, evecs, time_indices, num_nodes=None):
        B, P = time_indices.shape
        device = time_indices.device
        preds = torch.zeros((B, P), device=device)
        for t in range(P):
            prev = torch.zeros_like(preds)
            prev[:, 1:] = preds[:, :-1]
            out = self.forward(evals, evecs, time_indices, num_nodes=num_nodes, prev_betas=prev)
            preds[:, t] = out[:, t]
        return preds
