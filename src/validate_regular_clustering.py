"""Validate clustering of predicted beta-curves on large random d-regular graphs (N>20).

This script:
  1) Generates connected random d-regular graphs with N nodes.
  2) Runs a *size-generalizable* predictor (default: GNN) to produce beta curves.
  3) Clusters the predicted curves (hierarchical clustering) and saves overlay/PCA plots.
  4) Computes empirical spectral density of the scaled adjacency and overlays a Wigner semicircle
     reference (heuristic) to support the random-matrix interpretation.

Notes on theory:
  - For random d-regular graphs, the rigorous limiting spectral density is the Kesten–McKay law.
  - After scaling (and for moderately large d), the bulk often looks close to a semicircle.

Example:
  python -m src.validate_regular_clustering \
    --n 24 --d 3 --num_graphs 200 \
    --model_type gnn --model_path models/checkpoints/gnn_model_tmp.pth
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def _ensure_connected_random_regular_graph(n: int, d: int, seed: int | None) -> nx.Graph:
    if d >= n:
        raise ValueError(f"random d-regular requires d < n, got d={d}, n={n}")
    if (n * d) % 2 != 0:
        raise ValueError(f"n*d must be even for a d-regular graph, got n={n}, d={d}")

    s = seed
    for _ in range(10_000):
        g = nx.random_regular_graph(d, n, seed=s)
        if nx.is_connected(g):
            return g
        s = None if s is None else s + 1
    raise RuntimeError("Failed to sample a connected random regular graph after many tries")


def _wigner_semicircle_density(x: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Wigner semicircle density with support [-R, R]."""
    y = np.zeros_like(x, dtype=np.float64)
    inside = np.abs(x) <= radius
    y[inside] = (2.0 / (np.pi * radius**2)) * np.sqrt(radius**2 - x[inside] ** 2)
    return y


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """Return 2D PCA coordinates via SVD (no sklearn dependency)."""
    x0 = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(x0, full_matrices=False)
    return u[:, :2] * s[:2]


def _fiedler_feature_from_adj(adj: np.ndarray) -> np.ndarray:
    """Compute a 1D spectral node feature from the (combinatorial) Laplacian.

    Uses the Fiedler vector (2nd smallest eigenvector) of L = D - A.
    Sign is fixed deterministically to reduce sign-flip ambiguity.

    Returns shape [N] float64.
    """
    a = adj.astype(np.float64)
    deg = a.sum(axis=1)
    l = np.diag(deg) - a
    # For these N (~20-100), dense eigh is fine.
    w, v = np.linalg.eigh(l)
    if v.shape[1] < 2:
        feat = v[:, 0]
    else:
        feat = v[:, 1]
    # Fix sign
    if float(feat.sum()) < 0:
        feat = -feat
    # Normalize variance for numerical stability
    feat = feat - feat.mean()
    std = feat.std()
    if std > 1e-12:
        feat = feat / std
    return feat


@dataclass
class Predictions:
    betas: np.ndarray  # [G, T]
    labels: np.ndarray  # [G]
    eigvals_scaled: np.ndarray  # concatenated eigenvalues across graphs


def _load_model(model_type: str, model_path: str, device: torch.device):
    if model_type == "gnn":
        from src.models.gnn import FALQONGNN

        # max_nodes here is only a training-time pad size; the model itself is size-tolerant.
        model = FALQONGNN(max_nodes=12, output_len=30)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        return model.to(device).eval()

    if model_type == "transformer":
        from src.models.transformer import FALQONTransformer

        # WARNING: this transformer flattens adjacency => input dim depends on max_nodes.
        # Only works if the checkpoint was trained with max_nodes >= N.
        model = FALQONTransformer(max_nodes=12, output_len=30, d_model=64)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        return model.to(device).eval()

    raise ValueError(f"Unknown model_type={model_type}. Use gnn|transformer")


@torch.no_grad()
def _predict_betas(
    model_type: str,
    model,
    adj: np.ndarray,
    device: torch.device,
    gnn_feature: str = "ones",
) -> np.ndarray:
    n = adj.shape[0]
    if model_type == "gnn":
        adj_t = torch.tensor(adj, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.ones((1, n), dtype=torch.float32, device=device)
        node_feat_t = None
        if gnn_feature == "fiedler":
            feat = _fiedler_feature_from_adj(adj)
            node_feat_t = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N]
        elif gnn_feature != "ones":
            raise ValueError(f"Unknown gnn_feature={gnn_feature}. Use ones|fiedler")

        out = model(adj_t, mask_t, node_features=node_feat_t)
        return out.squeeze(0).detach().cpu().numpy().astype(np.float64)

    if model_type == "transformer":
        max_nodes = getattr(model, "max_nodes", None)
        if max_nodes is None:
            raise RuntimeError("Transformer model missing max_nodes")
        if n > int(max_nodes):
            raise ValueError(
                f"Transformer checkpoint max_nodes={max_nodes} < n={n}. "
                "You must retrain with larger max_nodes or use --model_type gnn."
            )
        padded = np.zeros((int(max_nodes), int(max_nodes)), dtype=np.float32)
        padded[:n, :n] = adj.astype(np.float32)
        adj_t = torch.tensor(padded, dtype=torch.float32, device=device).unsqueeze(0)
        out = model(adj_t, mask=None)
        return out.squeeze(0).detach().cpu().numpy().astype(np.float64)

    raise ValueError(model_type)


def run(args: argparse.Namespace) -> Predictions:
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if args.device in {"cpu", "cuda"}:
        device = torch.device(args.device)

    model = _load_model(args.model_type, args.model_path, device)

    betas_list: list[np.ndarray] = []
    eigvals_all: list[np.ndarray] = []

    for i in range(args.num_graphs):
        g = _ensure_connected_random_regular_graph(args.n, args.d, None if args.seed is None else args.seed + i)
        adj = nx.to_numpy_array(g, dtype=np.float32)

        betas = _predict_betas(args.model_type, model, adj, device, gnn_feature=args.gnn_feature)
        betas_list.append(betas)

        # Spectrum: scale adjacency so the nontrivial bulk lies roughly in [-2, 2]
        # for d-regular graphs: bulk support approx [-2*sqrt(d-1), 2*sqrt(d-1)]
        scale = np.sqrt(max(args.d - 1, 1))
        a_scaled = adj / scale
        w = np.linalg.eigvalsh(a_scaled)
        # remove the trivial top eigenvalue ~ d/scale (dominant regularity mode)
        w = np.sort(w)[:-1]
        eigvals_all.append(w.astype(np.float64))

    betas_mat = np.stack(betas_list, axis=0)  # [G, T]
    eigvals_scaled = np.concatenate(eigvals_all, axis=0)

    # Clustering on curves
    dist = pdist(betas_mat, metric=args.metric)
    z = linkage(dist, method=args.linkage)
    labels = fcluster(z, t=args.num_clusters, criterion="maxclust")

    return Predictions(betas=betas_mat, labels=labels.astype(np.int32), eigvals_scaled=eigvals_scaled)


def _plot_curves(out_dir: str, betas: np.ndarray, labels: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)

    t = np.arange(betas.shape[1])
    k = int(labels.max())

    # Use stable, distinct colors for clusters
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(11, 6))
    for c in range(1, k + 1):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        color = cmap((c - 1) % 10)
        for j in idx[: min(len(idx), 50)]:
            plt.plot(t, betas[j], alpha=0.12, color=color)
        mean_curve = betas[idx].mean(axis=0)
        plt.plot(
            t,
            mean_curve,
            linewidth=3.5,
            color=color,
            label=f"cluster {c} (n={len(idx)})",
            zorder=10 + c,
        )

    plt.title("Predicted beta curves on random regular graphs")
    plt.xlabel("layer step")
    plt.ylabel("beta")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_overlay.png"), dpi=200)
    plt.close()

    # Zoomed-in overlay (the large early spike can hide small between-cluster differences)
    if betas.shape[1] >= 6:
        plt.figure(figsize=(11, 6))
        t2 = t[2:]
        # robust y-limits based on percentiles
        y = betas[:, 2:].reshape(-1)
        lo, hi = np.percentile(y, [1, 99])
        pad = 0.1 * max(1e-6, hi - lo)
        for c in range(1, k + 1):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            color = cmap((c - 1) % 10)
            mean_curve = betas[idx].mean(axis=0)
            plt.plot(t2, mean_curve[2:], linewidth=3.5, color=color, label=f"cluster {c}")
        plt.ylim(lo - pad, hi + pad)
        plt.title("Cluster mean beta curves (zoomed, steps >= 2)")
        plt.xlabel("layer step")
        plt.ylabel("beta")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curves_overlay_zoom.png"), dpi=200)
        plt.close()

    # Difference-to-global-mean plot to visualize tiny separations
    global_mean = betas.mean(axis=0)
    plt.figure(figsize=(11, 6))
    for c in range(1, k + 1):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        color = cmap((c - 1) % 10)
        mean_curve = betas[idx].mean(axis=0)
        plt.plot(t, mean_curve - global_mean, linewidth=3.0, color=color, label=f"cluster {c}")
    plt.title("Cluster mean minus global mean (highlights subtle differences)")
    plt.xlabel("layer step")
    plt.ylabel(r"$\Delta$ beta")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_means_delta.png"), dpi=200)
    plt.close()

    coords = _pca_2d(betas)
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=18, cmap="tab10", alpha=0.9)
    plt.title("PCA of predicted curves")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=200)
    plt.close()


def _plot_spectrum(out_dir: str, eigvals_scaled: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    bins = 80
    plt.hist(eigvals_scaled, bins=bins, density=True, alpha=0.55, label="empirical bulk eigenvalues")

    xs = np.linspace(-2.0, 2.0, 400)
    ys = _wigner_semicircle_density(xs, radius=2.0)
    plt.plot(xs, ys, "k-", linewidth=2.5, label="Wigner semicircle (reference)")

    plt.title("Scaled adjacency spectrum (bulk) vs semicircle")
    plt.xlabel(r"$\lambda$ of $A/\sqrt{d-1}$ (largest eigenvalue removed)")
    plt.ylabel("density")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectrum_density.png"), dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24, help="number of nodes (recommend > 20)")
    p.add_argument("--d", type=int, default=3, help="regular degree")
    p.add_argument("--num_graphs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--model_type", type=str, default="gnn", choices=["gnn", "transformer"])
    p.add_argument("--model_path", type=str, default="models/checkpoints/gnn_model_tmp.pth")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument(
        "--gnn_feature",
        type=str,
        default="ones",
        choices=["ones", "fiedler"],
        help="GNN node feature. 'ones' may collapse on d-regular graphs; 'fiedler' breaks symmetry.",
    )

    p.add_argument("--num_clusters", type=int, default=3)
    p.add_argument("--metric", type=str, default="euclidean", help="pdist metric")
    p.add_argument("--linkage", type=str, default="average", choices=["single", "complete", "average", "ward"])

    p.add_argument("--out_dir", type=str, default="output/regular_clustering")

    args = p.parse_args()

    preds = run(args)

    # Warn if all curves are identical (common on regular graphs with constant node features)
    rounded = np.round(preds.betas, 6)
    uniq = len({tuple(row) for row in rounded})
    if uniq <= 1 and args.model_type == "gnn" and args.gnn_feature == "ones":
        print(
            "⚠️  All predicted curves are identical. This is expected for GCNs with constant node features on d-regular graphs.\n"
            "    Try: --gnn_feature fiedler (spectral node feature) to break symmetry."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        os.path.join(args.out_dir, "results.npz"),
        betas=preds.betas,
        labels=preds.labels,
        eigvals_scaled=preds.eigvals_scaled,
        n=args.n,
        d=args.d,
        num_graphs=args.num_graphs,
    )

    _plot_curves(args.out_dir, preds.betas, preds.labels)
    _plot_spectrum(args.out_dir, preds.eigvals_scaled)

    print(f"✅ saved: {args.out_dir}/curves_overlay.png")
    print(f"✅ saved: {args.out_dir}/curves_overlay_zoom.png")
    print(f"✅ saved: {args.out_dir}/cluster_means_delta.png")
    print(f"✅ saved: {args.out_dir}/pca_scatter.png")
    print(f"✅ saved: {args.out_dir}/spectrum_density.png")
    print(f"✅ saved: {args.out_dir}/results.npz")


if __name__ == "__main__":
    main()
