import torch
import torch.nn as nn


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # h: [B, N, Din]
        # adj_norm: [B, N, N]
        h = torch.bmm(adj_norm, h)
        return self.lin(h)


class FALQONGNN(nn.Module):
    """A lightweight GNN baseline that predicts the beta sequence from an adjacency matrix.

    No external GNN libraries required (torch_geometric/dgl).

    Inputs:
      - adj:  [B, N, N] padded adjacency (float)
      - mask: [B, N] node mask (float/bool), 1 for real nodes

    Output:
      - betas: [B, output_len]
    """

    def __init__(
        self,
        max_nodes: int = 12,
        output_len: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.output_len = output_len

        # Node features: a single constant feature per node (can be extended later)
        in_dim = 1

        layers = []
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(SimpleGCNLayer(dims[i], dims[i + 1]))
        self.gcn_layers = nn.ModuleList(layers)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len),
        )

    @staticmethod
    def _normalize_adj(adj: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # Add self-loops
        bsz, n, _ = adj.shape
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(bsz, -1, -1)
        a = adj + eye

        if mask is not None:
            if mask.dtype != adj.dtype:
                mask_f = mask.to(dtype=adj.dtype)
            else:
                mask_f = mask
            # Zero-out padded rows/cols (keeps normalization stable)
            m = mask_f.unsqueeze(2) * mask_f.unsqueeze(1)  # [B, N, N]
            a = a * m

        deg = a.sum(dim=-1)  # [B, N]
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        d_left = deg_inv_sqrt.unsqueeze(-1)
        d_right = deg_inv_sqrt.unsqueeze(-2)
        return a * d_left * d_right

    def forward(
        self,
        adj: torch.Tensor,
        mask: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # adj: [B, N, N]
        bsz, n, _ = adj.shape
        if n != self.max_nodes:
            # evaluation code pads to max_nodes, but be tolerant
            self.max_nodes = n

        if mask is None:
            mask_f = None
        else:
            mask_f = mask

        adj_norm = self._normalize_adj(adj, mask_f)

        # Node features
        if node_features is None:
            # Default: constant feature per node.
            h = torch.ones((bsz, n, 1), device=adj.device, dtype=adj.dtype)
        else:
            # Expect shape [B, N] or [B, N, 1]
            if node_features.dim() == 2:
                h = node_features.unsqueeze(-1)
            else:
                h = node_features
            if h.shape[:2] != (bsz, n):
                raise ValueError(f"node_features shape {tuple(h.shape)} incompatible with adj {(bsz, n, n)}")
            if h.shape[-1] != 1:
                raise ValueError("This baseline expects 1D node features (last dim = 1).")
            h = h.to(device=adj.device, dtype=adj.dtype)

        for layer in self.gcn_layers:
            h = layer(h, adj_norm)
            h = self.act(h)
            h = self.dropout(h)

        # Masked mean pooling
        if mask_f is None:
            g = h.mean(dim=1)
        else:
            if mask_f.dtype != h.dtype:
                mask_f2 = mask_f.to(dtype=h.dtype)
            else:
                mask_f2 = mask_f
            denom = mask_f2.sum(dim=1, keepdim=True).clamp(min=1.0)
            g = (h * mask_f2.unsqueeze(-1)).sum(dim=1) / denom

        return self.readout(g)
