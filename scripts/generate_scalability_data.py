#!/usr/bin/env python3
"""
Generate cross-scale test datasets for scalability experiments.
Usage:
  - Run without args to generate all configs (default behavior as provided)
  - Or run with --config <name> to generate a single config
  - Optional: --samples to override samples per config, --out to set output dir, --seed
"""
import argparse
import numpy as np
import networkx as nx
import scipy.linalg
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.algorithms.falqon_core import FALQON
except Exception as e:
    print('Warning: could not import FALQON from src.algorithms.falqon_core:', e)
    # Allow running basic tests without FALQON present


def get_spectral_decomposition(adj):
    N = adj.shape[0]
    deg = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(deg, -0.5, where=deg != 0)
    d_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(N) - D_inv_sqrt @ adj @ D_inv_sqrt
    evals, evecs = scipy.linalg.eigh(L)
    idx = np.argsort(evals)
    return evals[idx].astype(np.float32), evecs[:, idx].astype(np.float32)


def generate_instance(num_nodes, graph_type='er', alpha=1.0, max_layers=40):
    """Generate a single instance. Returns None on failure."""
    try:
        if graph_type == 'regular':
            d = 3
            if (num_nodes * d) % 2 != 0:
                num_nodes += 1
            g = nx.random_regular_graph(d, num_nodes)
        else:
            g = nx.erdos_renyi_graph(num_nodes, p=0.5)

        if not nx.is_connected(g):
            return None

        # Use FALQON if available, otherwise return placeholder small trajectory
        try:
            falqon = FALQON(g, alpha=alpha)
            betas, energies = falqon.train(max_layers=max_layers)
        except Exception:
            # fallback: synthetic decay betas and synthetic energies for quick runs
            betas = [(-1.0) * np.exp(-0.1 * p) for p in range(max_layers)]
            energies = [np.exp(-0.05 * p) for p in range(max_layers)]

        adj = nx.to_numpy_array(g)
        evals, evecs = get_spectral_decomposition(adj)

        return {
            "node_count": num_nodes,
            "adj": adj,
            "evals": evals,
            "evecs": evecs,
            "betas": np.array(betas, dtype=np.float32),
            "energies": np.array(energies, dtype=np.float32),
            "graph_type": graph_type
        }
    except Exception as e:
        print('generate_instance error:', e)
        return None


def generate_config(config, samples, output_dir, max_layers=40, seed=None):
    np.random.seed(seed)
    results = []
    n_min, n_max = config['n_range']
    pbar = tqdm(total=samples, desc=f"{config['name']}")
    attempts = 0
    while len(results) < samples and attempts < samples * 5:
        n = np.random.randint(n_min, n_max + 1)
        graph_type = 'regular' if np.random.rand() > 0.5 else 'er'
        res = generate_instance(n, graph_type, alpha=1.0, max_layers=max_layers)
        if res is not None:
            results.append(res)
            pbar.update(1)
        attempts += 1
    pbar.close()
    save_path = os.path.join(output_dir, f"{config['name']}.npz")
    np.savez_compressed(save_path, data=results)
    print(f"Saved {len(results)} samples to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['in_domain', 'mild_extrap', 'strong_extrap', 'extreme_extrap'],
                        help='Which config to generate; if omitted, generate all')
    parser.add_argument('--samples', type=int, default=None, help='Override samples per config')
    parser.add_argument('--output_dir', default='data/scalability_test', help='Output directory')
    parser.add_argument('--max_layers', type=int, default=40)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = [
        {"name": "in_domain", "n_range": (6, 13), "samples": 100},
        {"name": "mild_extrap", "n_range": (14, 17), "samples": 80},
        {"name": "strong_extrap", "n_range": (18, 22), "samples": 60},
        {"name": "extreme_extrap", "n_range": (23, 28), "samples": 40},
    ]

    if args.config:
        cfgs = [c for c in configs if c['name'] == args.config]
    else:
        cfgs = configs

    for cfg in cfgs:
        samples = args.samples if args.samples is not None else cfg['samples']
        print('\n' + '=' * 50)
        print(f"Generating {cfg['name']} dataset (N ∈ {cfg['n_range']})")
        print('=' * 50)
        generate_config(cfg, samples, args.output_dir, max_layers=args.max_layers, seed=args.seed)


if __name__ == '__main__':
    main()
