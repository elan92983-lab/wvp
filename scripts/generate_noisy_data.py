#!/usr/bin/env python3
"""
Generate noisy FALQON trajectory datasets with configurable noise levels.
Accepts CLI args to select specific noise level, sample count and node range.
"""
import argparse
import numpy as np
import networkx as nx
import scipy.linalg
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Noisy FALQON implementation (simplified) ---
class NoisyFALQON:
    """
    A simplified noisy FALQON simulator (suitable for small n)
    """
    def __init__(self, graph, alpha=1.0, 
                 shot_noise_std=0.0, decoherence_rate=0.0, gate_error_prob=0.0):
        self.graph = graph
        self.n_qubits = graph.number_of_nodes()
        self.alpha = alpha
        self.shot_noise_std = shot_noise_std
        self.decoherence_rate = decoherence_rate
        self.gate_error_prob = gate_error_prob
        self._build_hamiltonians()

    def _build_hamiltonians(self):
        n = self.n_qubits
        dim = 2**n
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        def tensor_op(op, i, n):
            result = np.array([[1]], dtype=complex)
            for j in range(n):
                result = np.kron(result, op if j == i else I)
            return result

        self.H_D = np.zeros((dim, dim), dtype=complex)
        for i in range(n):
            self.H_D += tensor_op(X, i, n)

        self.H_P = np.zeros((dim, dim), dtype=complex)
        for i, j in self.graph.edges():
            ZiZj = tensor_op(Z, i, n) @ tensor_op(Z, j, n)
            self.H_P += 0.5 * (np.eye(dim) - ZiZj)

        self.A = 1j * (self.H_D @ self.H_P - self.H_P @ self.H_D)

    def _apply_decoherence(self, state, dt=1.0):
        if self.decoherence_rate <= 0:
            return state
        decay = np.exp(-self.decoherence_rate * dt)
        noise = (1 - decay) * np.random.randn(len(state)) * 0.01
        state = state * decay + noise
        state = state / np.linalg.norm(state)
        return state

    def _apply_gate_error(self, state):
        if self.gate_error_prob <= 0 or np.random.rand() > self.gate_error_prob:
            return state
        noise = np.random.randn(len(state)) * 0.01
        state = state + noise
        state = state / np.linalg.norm(state)
        return state

    def train(self, max_layers=40):
        dim = 2**self.n_qubits
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)
        betas = []
        energies = []
        for p in range(max_layers):
            A_expect = np.real(np.conj(state) @ self.A @ state)
            if self.shot_noise_std > 0:
                A_expect += np.random.randn() * self.shot_noise_std
            beta = -self.alpha * A_expect
            betas.append(beta)

            U_P = scipy.linalg.expm(-1j * self.H_P)
            U_D = scipy.linalg.expm(-1j * beta * self.H_D)
            state = U_D @ U_P @ state

            state = self._apply_decoherence(state)
            state = self._apply_gate_error(state)
            state = state / np.linalg.norm(state)

            energy = np.real(np.conj(state) @ self.H_P @ state)
            energies.append(energy)
        return betas, energies


# --- utilities ---

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


def generate_noisy_instance(num_nodes, noise_config, graph_type='er', max_layers=40):
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

        falqon_clean = NoisyFALQON(g, alpha=1.0)
        betas_clean, energies_clean = falqon_clean.train(max_layers=max_layers)

        falqon_noisy = NoisyFALQON(
            g, alpha=1.0,
            shot_noise_std=noise_config['shot_noise'],
            decoherence_rate=noise_config['decoherence'],
            gate_error_prob=noise_config['gate_error']
        )
        betas_noisy, energies_noisy = falqon_noisy.train(max_layers=max_layers)

        adj = nx.to_numpy_array(g)
        evals, evecs = get_spectral_decomposition(adj)

        return {
            "node_count": num_nodes,
            "adj": adj,
            "evals": evals,
            "evecs": evecs,
            "betas_clean": np.array(betas_clean, dtype=np.float32),
            "betas_noisy": np.array(betas_noisy, dtype=np.float32),
            "energies_clean": np.array(energies_clean, dtype=np.float32),
            "energies_noisy": np.array(energies_noisy, dtype=np.float32),
            "graph_type": graph_type,
            "noise_config": noise_config
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


# --- main with CLI ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', choices=['no_noise','low_noise','medium_noise','high_noise','extreme_noise','all'], default='all')
    parser.add_argument('--samples-per-config', type=int, default=50)
    parser.add_argument('--n-min', type=int, default=6)
    parser.add_argument('--n-max', type=int, default=10)
    parser.add_argument('--output-dir', default='data/noise_test')
    parser.add_argument('--max-layers', type=int, default=40)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    noise_configs = [
        {"name": "no_noise", "shot_noise": 0.0, "decoherence": 0.0, "gate_error": 0.0},
        {"name": "low_noise", "shot_noise": 0.05, "decoherence": 0.01, "gate_error": 0.001},
        {"name": "medium_noise", "shot_noise": 0.1, "decoherence": 0.02, "gate_error": 0.005},
        {"name": "high_noise", "shot_noise": 0.2, "decoherence": 0.05, "gate_error": 0.01},
        {"name": "extreme_noise", "shot_noise": 0.3, "decoherence": 0.1, "gate_error": 0.02},
    ]

    if args.noise != 'all':
        noise_configs = [nc for nc in noise_configs if nc['name'] == args.noise]

    samples_per_config = args.samples_per_config
    n_min, n_max = args.n_min, args.n_max

    np.random.seed(args.seed)

    for noise_config in noise_configs:
        print(f"\n{'='*50}")
        print(f"Generating {noise_config['name']} (Shot={noise_config['shot_noise']}, Decoh={noise_config['decoherence']}, Gate={noise_config['gate_error']})")
        print(f"{'='*50}")

        results = []
        pbar = tqdm(total=samples_per_config)
        attempts = 0

        while len(results) < samples_per_config and attempts < samples_per_config * 3:
            n = np.random.randint(n_min, n_max + 1)
            graph_type = 'regular' if np.random.rand() > 0.5 else 'er'
            res = generate_noisy_instance(n, noise_config, graph_type, max_layers=args.max_layers)
            if res is not None:
                results.append(res)
                pbar.update(1)
            attempts += 1

        pbar.close()
        save_path = os.path.join(output_dir, f"{noise_config['name']}.npz")
        np.savez_compressed(save_path, data=results)
        print(f"Saved {len(results)} samples to {save_path}")