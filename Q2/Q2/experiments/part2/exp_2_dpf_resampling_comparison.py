"""TASKS.md 2-2-ii: Compare DPF resampling strategies on proposal learning.

6 methods from Chen et al. (2023) compared on Corenflos(2021) Section 5.2:
- OT (Sinkhorn DET) -- fully differentiable transport [Chen(23) 5.2]
- Soft resampling -- partial diff through weight correction [Chen(23) 5.1]
- StopGrad -- biased gradient, stop_gradient at resample [Chen(23) p.8]
- WeightPres -- weight mean preserves gradient path [Chen(23) p.9, ref 73]
- Truncated -- gradient blocked at every timestep [Chen(23) p.8, ref 47]
- NoResample -- no resampling, full gradient but ESS collapses [Chen(23) p.8-9]

Metrics:
- Accuracy: RMSE(phi-1), ESS
- Differentiability: gradient norm mean/std
- Efficiency: runtime per step

All methods at N=25, B=4 for fair comparison.

Usage:
    cd Q2 && python experiments/part2/exp_2_dpf_resampling_comparison.py
"""
import os
import sys
import time

import numpy as np
import json

import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm import CorenflosLGSSM
from src.resampling import (SinkhornOTResampler, SoftResampler, MultinomialResampler,
                             WeightPreservationResampler, NoResampling)
from src.utils.config_manager import check_and_save

DTYPE = tf.float32
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part2', 'exp_dpf_resampling_comparison')

_LOG_2PI = tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))


def _log_gauss_diag(x, mean, diag_var):
    d = tf.cast(tf.shape(x)[-1], DTYPE)
    mahal = tf.reduce_sum((x - mean) ** 2 / diag_var, axis=-1)
    log_det = tf.reduce_sum(tf.math.log(diag_var))
    return -0.5 * (d * _LOG_2PI + log_det + mahal)


def _log_gauss_full(x, mean, cov_inv, log_det_cov):
    d = tf.cast(tf.shape(x)[-1], DTYPE)
    diff = x - mean
    mahal = tf.reduce_sum(diff * tf.linalg.matvec(cov_inv, diff), axis=-1)
    return -0.5 * (d * _LOG_2PI + log_det_cov + mahal)


def make_train_step(ssm, N, batch_size, resampler, lr, truncate_all=False):
    """Create @tf.function compiled train step."""
    d_x = ssm.state_dim
    d_y = ssm.obs_dim
    d_x_f = tf.constant(d_x, dtype=DTYPE)
    A = ssm.A
    H = ssm.H
    R_inv = ssm.R_inv
    log_det_R = ssm.log_det_R

    rng = tf.random.Generator.from_seed(0)

    @tf.function
    def train_step(log_phi_x, phi_y, ys, optimizer):
        T = tf.shape(ys)[0]
        N_f = tf.cast(N, DTYPE)
        T_f = tf.cast(T, DTYPE)

        with tf.GradientTape() as tape:
            phi_x = tf.exp(log_phi_x)
            sqrt_phi_x = tf.exp(log_phi_x / 2.0)
            inv_phi_x = tf.exp(-log_phi_x)
            Gamma = tf.linalg.diag(phi_y, num_rows=d_x)

            particles = rng.normal(shape=(batch_size, N, d_x), dtype=DTYPE)
            log_weights = tf.fill([batch_size, N], -tf.math.log(N_f))

            elbo = tf.zeros(batch_size, dtype=DTYPE)
            total_ess = tf.zeros(batch_size, dtype=DTYPE)
            ess_arr = tf.TensorArray(DTYPE, size=T, dynamic_size=False)

            for t in tf.range(T):
                y_t = ys[t]

                # Propose
                Ax = tf.linalg.matvec(A, particles)
                Gy = tf.linalg.matvec(Gamma, y_t)
                m_prop = (Ax + Gy) * inv_phi_x
                noise = rng.normal(shape=(batch_size, N, d_x), dtype=DTYPE)
                x_new = m_prop + noise * sqrt_phi_x

                # Weight
                Hx = tf.linalg.matvec(H, x_new)
                y_broad = tf.broadcast_to(y_t, tf.shape(Hx))
                log_obs = _log_gauss_full(y_broad, Hx, R_inv, log_det_R)

                diff_trans = x_new - Ax
                log_trans = -0.5 * (d_x_f * _LOG_2PI + tf.reduce_sum(diff_trans ** 2, axis=-1))

                log_prop = _log_gauss_diag(x_new, m_prop, phi_x)

                log_w_inc = log_obs + log_trans - log_prop
                log_w_total = log_weights + log_w_inc
                elbo = elbo + tf.reduce_logsumexp(log_w_total, axis=1)

                # ESS
                log_weights = log_w_total - tf.reduce_max(log_w_total, axis=1, keepdims=True)
                w = tf.nn.softmax(log_weights, axis=1)
                ess_t = 1.0 / tf.reduce_sum(w ** 2, axis=1)
                total_ess = total_ess + ess_t
                ess_arr = ess_arr.write(t, tf.reduce_mean(ess_t))

                # Resample when ESS < N/2
                particles_r, lw_r = resampler.apply(log_weights, x_new)
                if not resampler.DIFFERENTIABLE:
                    particles_r = tf.stop_gradient(particles_r)
                    lw_r = tf.stop_gradient(lw_r)

                flags = ess_t < 0.5 * N_f
                flags_p = flags[:, tf.newaxis, tf.newaxis]
                flags_w = flags[:, tf.newaxis]
                particles = tf.where(flags_p, particles_r, x_new)
                log_weights = tf.where(flags_w, lw_r, log_weights)

                # Truncated backprop
                if truncate_all:
                    particles = tf.stop_gradient(particles)
                    log_weights = tf.stop_gradient(log_weights)

            loss = -tf.reduce_mean(elbo) / T_f
            mean_ess = tf.reduce_mean(total_ess) / T_f

        grads = tape.gradient(loss, [log_phi_x, phi_y])

        # Gradient norms (for differentiability metric)
        grad_norm = tf.constant(0.0, dtype=DTYPE)
        valid = tf.constant(True)
        if grads[0] is not None:
            gn = tf.norm(grads[0])
            valid = ~tf.math.is_nan(gn)
            grad_norm = tf.where(valid, gn, 0.0)

        if valid:
            clipped = [tf.clip_by_value(g, -100.0, 100.0) for g in grads if g is not None]
            variables = [v for g, v in zip(grads, [log_phi_x, phi_y]) if g is not None]
            if clipped:
                optimizer.apply_gradients(zip(clipped, variables))

        return loss, mean_ess, ess_arr.stack(), grad_norm

    return train_step


def measure_grad_variance(log_phi_x, phi_y, ssm, ys_tf, N, batch_size,
                          resampler, truncate_all, n_seeds=10):
    """Measure gradient variance at fixed phi by varying only the random seed.

    This isolates the effect of the resampler on gradient quality,
    independent of the optimization trajectory.

    Parameters
    ----------
    log_phi_x, phi_y : tf.Variable (frozen during measurement)
    n_seeds : int
        Number of different random seeds to evaluate.

    Returns
    -------
    grad_mean, grad_std : float
        Mean and std of gradient norms across seeds.
    """
    # Create a fresh train_step that only computes gradient (no optimizer update)
    eval_fn = make_train_step(ssm, N, batch_size, resampler, 0.05, truncate_all)
    dummy_opt = tf.keras.optimizers.SGD(learning_rate=0.0)  # lr=0, no update

    grad_norms = []
    for s in range(n_seeds):
        _, _, _, gn = eval_fn(log_phi_x, phi_y, ys_tf, dummy_opt)
        grad_norms.append(float(gn))

    return np.mean(grad_norms), np.std(grad_norms)


def train_one(ssm, ys_tf, N, resampler, name, n_steps, lr, seed,
              batch_size, truncate_all=False, n_grad_seeds=10,
              optimizer_type='adam'):
    """Train proposal and return metrics."""
    d_x = ssm.state_dim
    d_y = ssm.obs_dim

    log_phi_x = tf.Variable(tf.ones(d_x, dtype=DTYPE))
    phi_y = tf.Variable(tf.zeros(d_y, dtype=DTYPE))
    if optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_step_fn = make_train_step(ssm, N, batch_size, resampler, lr, truncate_all)

    # Per-step history: ELBO, ESS, RMSE, grad_norm, time
    history = {'step': [], 'elbo': [], 'ess': [], 'rmse': [], 'grad_norm': [], 'time': []}

    for step in range(n_steps):
        t0 = time.perf_counter()
        loss, mean_ess, ess_per_t, grad_norm = train_step_fn(
            log_phi_x, phi_y, ys_tf, optimizer)
        dt = time.perf_counter() - t0

        phi_all = tf.concat([tf.exp(log_phi_x), phi_y], axis=0)
        phi_err = float(tf.sqrt(tf.reduce_mean((phi_all - 1.0) ** 2)))

        history['step'].append(step)
        history['elbo'].append(float(-loss))      # ELBO/T (positive)
        history['ess'].append(float(mean_ess))
        history['rmse'].append(phi_err)
        history['grad_norm'].append(float(grad_norm))
        history['time'].append(dt)

        if step % 20 == 0 or step == n_steps - 1:
            print(f"    [{name}] step {step:3d}  ELBO/T={float(-loss):.2f}  "
                  f"RMSE={phi_err:.4f}  ESS={float(mean_ess):.1f}  "
                  f"grad={float(grad_norm):.4f}  time={dt:.2f}s", flush=True)

    # Measure gradient variance at converged phi (fixed phi, varying seeds)
    grad_mean, grad_std = measure_grad_variance(
        log_phi_x, phi_y, ssm, ys_tf, N, batch_size,
        resampler, truncate_all, n_seeds=n_grad_seeds)
    print(f"    [{name}] grad variance: mean={grad_mean:.4f} std={grad_std:.4f}", flush=True)

    return {
        'rmse': history['rmse'][-1],
        'ess': history['ess'][-1],
        'grad_mean': grad_mean,
        'grad_std': grad_std,
        'time_mean': np.mean(history['time'][1:]),
        'history': history,  # per-step data for plotting
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--M', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default: 0.1 for SGD, 0.05 for Adam.')
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.1 if args.optimizer == 'sgd' else 0.05

    results_dir = os.path.join(RESULTS_DIR, f'{args.optimizer}_lr{args.lr}_eps{args.epsilon}')

    d_x, d_y, T = 25, 1, 100
    N, B = 25, 4
    n_steps = 100
    lr = args.lr
    M = args.M
    seed = args.seed

    ssm = CorenflosLGSSM(d_x=d_x, d_y=d_y, dtype=DTYPE)

    # (name, resampler, truncate_all)
    methods = [
        ('OT',          SinkhornOTResampler(epsilon=args.epsilon, scaling=0.9), False),
        ('Soft',        SoftResampler(alpha=0.5),                      False),
        ('StopGrad',    MultinomialResampler(),                        False),
        ('WeightPres',  WeightPreservationResampler(),                 False),
        ('Truncated',   MultinomialResampler(),                        True),
        ('NoResample',  NoResampling(),                                False),
    ]

    config = {
        'd_x': d_x, 'd_y': d_y, 'T': T,
        'N': N, 'batch_size': B, 'n_steps': n_steps,
        'optimizer': args.optimizer, 'lr': lr,
        'M': M, 'seed': seed,
        'epsilon': args.epsilon,
        'methods': [name for name, _, _ in methods],
    }
    path, is_dup = check_and_save(results_dir, config)
    if is_dup:
        print(f"Duplicate run found at {path}, skipping.")
        return

    print(f"=== 6-Method DPF Resampling Comparison ===")
    print(f"d_x={d_x}, d_y={d_y}, T={T}, N={N}, B={B}, steps={n_steps}, lr={lr}, M={M}")
    print(f"Methods: {[m[0] for m in methods]}\n")

    all_results = {name: {'rmse': [], 'ess': [], 'grad_mean': [], 'grad_std': [], 'time': []}
                   for name, _, _ in methods}
    method_histories = {name: [] for name, _, _ in methods}

    for m in range(M):
        rng_np = np.random.default_rng(seed + m)
        _, ys = ssm.simulate(T, rng_np)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        print(f"--- Realization {m+1}/{M} ---")

        for name, resampler, truncate_all in methods:
            result = train_one(ssm, ys_tf, N, resampler, name, n_steps, lr,
                               seed=seed + m * 10000, batch_size=B,
                               truncate_all=truncate_all,
                               optimizer_type=args.optimizer)

            all_results[name]['rmse'].append(result['rmse'])
            all_results[name]['ess'].append(result['ess'])
            all_results[name]['grad_mean'].append(result['grad_mean'])
            all_results[name]['grad_std'].append(result['grad_std'])
            all_results[name]['time'].append(result['time_mean'])

            h = np.column_stack([result['history']['elbo'], result['history']['ess'],
                                 result['history']['rmse'], result['history']['grad_norm'],
                                 result['history']['time']])
            method_histories[name].append(h)

        # Print per-realization summary
        for name, _, _ in methods:
            r = all_results[name]
            print(f"  {name:12s}: RMSE={r['rmse'][-1]:.4f}  ESS={r['ess'][-1]:.1f}  "
                  f"grad={r['grad_mean'][-1]:.4f}+/-{r['grad_std'][-1]:.4f}  "
                  f"time={r['time'][-1]:.3f}s", flush=True)
        print()

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary (M={M})")
    print(f"{'='*80}")
    print(f"{'Method':12s}  {'RMSE':>12s}  {'ESS':>8s}  {'ESS%':>6s}  "
          f"{'Grad Mean':>10s}  {'Grad Std':>10s}  {'Time/step':>10s}")
    print("-" * 80)
    for name, _, _ in methods:
        r = all_results[name]
        valid_rmse = [x for x in r['rmse'] if not np.isnan(x)]
        print(f"{name:12s}  "
              f"{np.mean(valid_rmse):12.4f}  "
              f"{np.mean(r['ess']):8.1f}  "
              f"{np.mean(r['ess'])/N*100:5.0f}%  "
              f"{np.mean(r['grad_mean']):10.4f}  "
              f"{np.mean(r['grad_std']):10.4f}  "
              f"{np.mean(r['time']):10.4f}s")

    # Save per-method NPZ
    for name, _, _ in methods:
        np.savez(os.path.join(results_dir, f'{name}.npz'),
                 steps=np.arange(n_steps),
                 histories=np.stack(method_histories[name]))

    # Save summary
    summary = {}
    for name, _, _ in methods:
        r = all_results[name]
        summary[name] = {
            'rmse_mean': float(np.nanmean(r['rmse'])),
            'rmse_std': float(np.nanstd(r['rmse'])),
            'ess_mean': float(np.mean(r['ess'])),
            'ess_pct': float(np.mean(r['ess']) / N * 100),
            'grad_mean': float(np.mean(r['grad_mean'])),
            'grad_std': float(np.mean(r['grad_std'])),
            'time_per_step': float(np.mean(r['time'])),
        }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {results_dir}/")


if __name__ == '__main__':
    main()
