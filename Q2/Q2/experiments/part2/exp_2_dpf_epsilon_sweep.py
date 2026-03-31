"""Part 2-2-1: OT epsilon sweep for bias-variance-speed tradeoff.

Runs proposal learning with OT resampler at different epsilon values.
Per-step ELBO/ESS/RMSE/grad_norm recorded for plotting convergence curves.

Bias:     large eps -> RMSE floor higher (diffuse transport)
Variance: small eps -> RMSE oscillates (noisy gradient)
Speed:    large eps -> fewer Sinkhorn iters -> faster per step

Usage:
    cd Q2 && python experiments/part2/exp_2_dpf_epsilon_sweep.py
    cd Q2 && python experiments/part2/exp_2_dpf_epsilon_sweep.py --N 50 --epsilons 0.5 2.0 --M 20
"""
import argparse
import os
import sys
import time

import numpy as np
import json

import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm import CorenflosLGSSM
from src.resampling import SinkhornOTResampler
from src.utils.config_manager import check_and_save
from experiments.part2.exp_2_dpf_resampling_comparison import make_train_step

DTYPE = tf.float32
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part2', 'exp_dpf_epsilon_sweep')


def train_one_eps(ssm, ys_tf, eps, N=25, batch_size=4, n_steps=100, lr=0.05,
                  max_iters=100):
    """Train proposal with given epsilon, return per-step history."""
    d_x = ssm.state_dim
    d_y = ssm.obs_dim

    log_phi_x = tf.Variable(tf.ones(d_x, dtype=DTYPE))
    phi_y = tf.Variable(tf.zeros(d_y, dtype=DTYPE))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    resampler = SinkhornOTResampler(epsilon=eps, scaling=0.9, max_iters=max_iters)
    train_step_fn = make_train_step(ssm, N, batch_size, resampler, lr, truncate_all=False)

    history = {'step': [], 'elbo': [], 'ess': [], 'rmse': [], 'grad_norm': [], 'time': []}

    for step in range(n_steps):
        t0 = time.perf_counter()
        loss, mean_ess, _, grad_norm = train_step_fn(log_phi_x, phi_y, ys_tf, optimizer)
        dt = time.perf_counter() - t0

        phi_all = tf.concat([tf.exp(log_phi_x), phi_y], axis=0)
        phi_err = float(tf.sqrt(tf.reduce_mean((phi_all - 1.0) ** 2)))

        history['step'].append(step)
        history['elbo'].append(float(-loss))
        history['ess'].append(float(mean_ess))
        history['rmse'].append(phi_err)
        history['grad_norm'].append(float(grad_norm))
        history['time'].append(dt)

    return history


def main():
    parser = argparse.ArgumentParser(description='OT epsilon sweep')
    parser.add_argument('--N', type=int, default=25, help='Number of particles')
    parser.add_argument('--B', type=int, default=4, help='Batch size')
    parser.add_argument('--n_steps', type=int, default=100, help='Gradient steps')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--M', type=int, default=30, help='Number of MC runs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epsilons', type=float, nargs='+',
                        default=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
                        help='Epsilon values to sweep')
    parser.add_argument('--max_iters', type=int, default=100,
                        help='Max Sinkhorn iterations (default: 100)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Subdirectory tag (e.g. N50)')
    args = parser.parse_args()

    d_x, d_y, T = 25, 1, 100
    N, B = args.N, args.B
    n_steps = args.n_steps
    lr = args.lr
    M = args.M
    seed = args.seed
    epsilons = args.epsilons

    out_dir = RESULTS_DIR
    if args.tag:
        out_dir = os.path.join(RESULTS_DIR, args.tag)

    config = {
        'd_x': d_x, 'd_y': d_y, 'T': T,
        'N': N, 'batch_size': B, 'n_steps': n_steps,
        'lr': lr, 'M': M, 'seed': seed, 'epsilons': epsilons,
        'max_iters': args.max_iters,
    }
    path, is_dup = check_and_save(out_dir, config)
    if is_dup:
        print(f"Duplicate run found at {path}, skipping.")
        return

    ssm = CorenflosLGSSM(d_x=d_x, d_y=d_y, dtype=DTYPE)

    print(f"=== OT Epsilon Sweep (Bias-Variance-Speed) ===")
    print(f"epsilons={epsilons}, N={N}, B={B}, steps={n_steps}, lr={lr}, M={M}\n")

    summary = {}
    hist_dir = os.path.join(out_dir, 'histories')
    os.makedirs(hist_dir, exist_ok=True)

    for eps in epsilons:
        rmses, esss, times = [], [], []
        all_histories = []

        for m in range(M):
            rng_np = np.random.default_rng(seed + m)
            _, ys = ssm.simulate(T, rng_np)
            ys_tf = tf.constant(ys, dtype=DTYPE)

            hist = train_one_eps(ssm, ys_tf, eps, N, B, n_steps, lr,
                                max_iters=args.max_iters)

            h = np.column_stack([hist['elbo'], hist['ess'], hist['rmse'],
                                 hist['grad_norm'], hist['time']])
            all_histories.append(h)

            rmses.append(hist['rmse'][-1])
            esss.append(hist['ess'][-1])
            times.append(np.mean(hist['time'][1:]))

            print(f"  eps={eps:.2f} M={m+1}: RMSE={hist['rmse'][-1]:.4f}  "
                  f"ESS={hist['ess'][-1]:.1f}  time/step={times[-1]:.3f}s", flush=True)

        # Save per-epsilon NPZ
        np.savez(os.path.join(hist_dir, f'eps{eps}.npz'),
                 steps=np.arange(n_steps),
                 histories=np.stack(all_histories))

        summary[str(eps)] = {
            'rmse_mean': float(np.mean(rmses)),
            'rmse_std': float(np.std(rmses)),
            'ess_mean': float(np.mean(esss)),
            'time_per_step': float(np.mean(times)),
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"{'eps':>6s}  {'RMSE':>12s}  {'ESS':>8s}  {'Time/step':>10s}")
    print("-" * 45)
    for eps_str, row in summary.items():
        print(f"{float(eps_str):6.2f}  {row['rmse_mean']:8.4f}+/-{row['rmse_std']:.4f}  "
              f"{row['ess_mean']:8.1f}  {row['time_per_step']:10.4f}s")

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
