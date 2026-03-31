"""Corenflos(2021) Section 5.2 replication: Learning the Proposal Distribution.

SSM (Eq. 16-17):
    X_{t+1} | X_t=x  ~  N(A x, I_{d_x}),   A_{ij} = 0.42^{|i-j|+1}
    Y_t     | X_t=x  ~  N(I_{d_y,d_x} x, 0.1 I_{d_y})

Learnable proposal (below Eq. 17):
    q_phi(x_t | x_{t-1}, y_t) = N(Delta^{-1}(A x_{t-1} + Gamma y_t), Delta)
    Delta = diag(exp(log_phi_x)),  Gamma = diag_{d_x,d_y}(phi_y)
    Optimal: log_phi_x = 0 (Delta=I), phi_y = 1 (Gamma=I_{d_x,d_y})

Paper settings (Section 5.2):
    d_x=25, d_y=1, T=100, M=100 realizations
    100 steps SGD lr=0.1 (paper text)
    PF baseline: N=500, multinomial resampling, biased gradients
    DPF: N=25, epsilon=0.5, 4 independent filters (batch_size=4)
    Metric: RMSE(phi-1), ESS after convergence
    Paper result: DPF RMSE=0.11, PF RMSE=0.22, ESS DPF~60%, PF~25%

Author code reference: filterflow-master/scripts/global_optimal_proposal_variational.py

Usage:
    cd Q2 && python experiments/part2/exp_2_dpf_proposal_learning.py
"""
import os
import sys
import time

import numpy as np
import json

import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm import CorenflosLGSSM
from src.resampling import SinkhornOTResampler, MultinomialResampler
from src.utils.config_manager import check_and_save

DTYPE = tf.float32
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part2', 'exp_dpf_proposal_learning')

_LOG_2PI = tf.math.log(tf.constant(2.0 * np.pi, dtype=tf.float32))


# Gaussian log-density helpers
def _log_gauss_diag(x, mean, diag_var):
    """Log N(x; mean, diag(diag_var)).  x:[B,N,d], mean:[B,N,d], diag_var:[d]."""
    d = tf.cast(tf.shape(x)[-1], DTYPE)
    mahal = tf.reduce_sum((x - mean) ** 2 / diag_var, axis=-1)
    log_det = tf.reduce_sum(tf.math.log(diag_var))
    return -0.5 * (d * _LOG_2PI + log_det + mahal)


def _log_gauss_full(x, mean, cov_inv, log_det_cov):
    """Log N(x; mean, cov).  x:[B,N,d], mean:[B,N,d] or broadcastable."""
    d = tf.cast(tf.shape(x)[-1], DTYPE)
    diff = x - mean
    mahal = tf.reduce_sum(diff * tf.linalg.matvec(cov_inv, diff), axis=-1)
    return -0.5 * (d * _LOG_2PI + log_det_cov + mahal)


# JIT-compiled filter + gradient step
def make_train_step(ssm, N, batch_size, resampler, lr):
    """Create a tf.function-compiled training step.

    Returns a function that takes (log_phi_x, phi_y, ys, optimizer)
    and returns (loss, mean_ess, ess_per_t, grad_norm).
    """
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

            loss = -tf.reduce_mean(elbo) / T_f
            mean_ess = tf.reduce_mean(total_ess) / T_f

        grads = tape.gradient(loss, [log_phi_x, phi_y])

        # Clip and apply
        valid = tf.reduce_all([
            grads[0] is not None,
            grads[1] is not None,
            ~tf.reduce_any(tf.math.is_nan(grads[0])),
            ~tf.reduce_any(tf.math.is_nan(grads[1])),
        ])
        if valid:
            clipped = [tf.clip_by_value(g, -100.0, 100.0) for g in grads]
            optimizer.apply_gradients(zip(clipped, [log_phi_x, phi_y]))

        # Gradient norm
        grad_norm = tf.constant(0.0, dtype=DTYPE)
        if grads[0] is not None:
            gn = tf.norm(grads[0])
            grad_norm = tf.where(~tf.math.is_nan(gn), gn, 0.0)

        return loss, mean_ess, ess_arr.stack(), grad_norm

    return train_step


# Training loop
def train_proposal(ssm, ys_tf, N, resampler, name,
                   n_steps=100, lr=0.1, seed=0,
                   batch_size=1, optimizer_type='adam'):
    """Train proposal via ELBO gradient ascent (JIT-compiled).

    Parameters
    ----------
    ssm : CorenflosLGSSM
    ys_tf : tf.Tensor [T, d_y]
    N : int
    resampler : ResamplerBase
    name : str
    n_steps : int
    lr : float
    seed : int
    batch_size : int

    Returns
    -------
    history : dict
    """
    d_x = ssm.state_dim
    d_y = ssm.obs_dim

    log_phi_x = tf.Variable(tf.ones(d_x, dtype=DTYPE))
    phi_y = tf.Variable(tf.zeros(d_y, dtype=DTYPE))
    if optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Create JIT-compiled train step (includes persistent RNG)
    train_step_fn = make_train_step(ssm, N, batch_size, resampler,
                                     lr)

    history = {'step': [], 'phi_rmse': [], 'elbo': [], 'ess': [],
               'grad_norm': [], 'time': [], 'ess_per_t': None}

    for step in range(n_steps):
        t0 = time.perf_counter()

        loss, mean_ess, ess_per_t, grad_norm = train_step_fn(
            log_phi_x, phi_y, ys_tf, optimizer)

        dt = time.perf_counter() - t0

        phi_x_val = tf.exp(log_phi_x)
        phi_all = tf.concat([phi_x_val, phi_y], axis=0)
        phi_err = float(tf.sqrt(tf.reduce_mean((phi_all - 1.0) ** 2)))

        history['step'].append(step)
        history['elbo'].append(float(-loss))        # ELBO/T (positive)
        history['ess'].append(float(mean_ess))
        history['phi_rmse'].append(phi_err)
        history['grad_norm'].append(float(grad_norm))
        history['time'].append(dt)

        if step % 10 == 0 or step == n_steps - 1:
            print(f"  [{name}] step {step:3d}  ELBO/T={float(-loss):8.2f}  "
                  f"RMSE={phi_err:.4f}  ESS={float(mean_ess):.1f}  "
                  f"grad={float(grad_norm):.4f}  time={dt:.1f}s", flush=True)

    history['ess_per_t'] = ess_per_t.numpy()
    return history


# Main
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Corenflos(2021) Sec 5.2 Proposal Learning')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default: 0.1 for SGD, 0.05 for Adam.')
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.5)
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.1 if args.optimizer == 'sgd' else 0.05

    # Results dir includes optimizer/lr/eps to avoid overwrite
    results_dir = os.path.join(RESULTS_DIR, f'{args.optimizer}_lr{args.lr}_eps{args.epsilon}')

    d_x, d_y, T = 25, 1, 100
    n_steps = 100
    epsilon = args.epsilon

    config = {
        'd_x': d_x, 'd_y': d_y, 'T': T,
        'n_steps': n_steps, 'epsilon': epsilon,
        'optimizer': args.optimizer, 'lr': args.lr,
        'M': args.M, 'seed': args.seed,
    }
    path, is_dup = check_and_save(results_dir, config)
    if is_dup:
        print(f"Duplicate run found at {path}, skipping.")
        return

    ssm = CorenflosLGSSM(d_x=d_x, d_y=d_y, dtype=DTYPE)

    print(f"Corenflos(2021) Section 5.2: Proposal Learning")
    print(f"optimizer={args.optimizer}, lr={args.lr}, M={args.M}, seed={args.seed}")
    print()

    methods = [
        ('DPF (N=25)',  SinkhornOTResampler(epsilon=epsilon, scaling=0.9), 25,  4),
        ('PF (N=500)',  MultinomialResampler(),                            500, 1),
    ]

    all_results = {name: {'phi_rmse': [], 'ess': []}
                   for name, _, _, _ in methods}
    method_histories = {name: [] for name, _, _, _ in methods}

    for m in range(args.M):
        rng_np = np.random.default_rng(args.seed + m)
        _, ys = ssm.simulate(T, rng_np)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        print(f"\n--- Realization {m+1}/{args.M} ---")

        for name, resampler, N, bs in methods:
            hist = train_proposal(
                ssm, ys_tf, N, resampler, name,
                n_steps=n_steps, lr=args.lr, seed=args.seed + m * 10000,
                batch_size=bs, optimizer_type=args.optimizer)

            all_results[name]['phi_rmse'].append(hist['phi_rmse'][-1])
            all_results[name]['ess'].append(hist['ess'][-1])

            h = np.column_stack([hist['elbo'], hist['ess'], hist['phi_rmse'],
                                 hist['grad_norm'], hist['time']])
            method_histories[name].append(h)

    print(f"\n{'='*60}")
    print(f"Summary over {args.M} realizations (paper: DPF=0.11, PF=0.22)")
    print(f"{'='*60}")
    print(f"{'Method':<15s}  {'RMSE(phi-1)':>12s}  {'ESS':>10s}")
    print("-" * 45)
    for name, _, _, _ in methods:
        r = all_results[name]
        rmse_vals = [v for v in r['phi_rmse'] if not np.isnan(v)]
        ess_vals = [v for v in r['ess'] if not np.isnan(v)]
        if rmse_vals:
            print(f"{name:<15s}  {np.mean(rmse_vals):12.4f}  {np.mean(ess_vals):10.1f}")
        else:
            print(f"{name:<15s}  {'NaN':>12s}  {'NaN':>10s}")

    # Save per-method NPZ
    for name, _, _, _ in methods:
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        np.savez(os.path.join(results_dir, f'{safe_name}.npz'),
                 steps=np.arange(n_steps),
                 histories=np.stack(method_histories[name]))

    # Save summary
    summary = {}
    for name, _, N, bs in methods:
        r = all_results[name]
        summary[name] = {
            'N': N, 'batch_size': bs,
            'optimizer': args.optimizer, 'lr': args.lr,
            'rmse_mean': float(np.nanmean(r['phi_rmse'])),
            'rmse_std': float(np.nanstd(r['phi_rmse'])),
            'ess_mean': float(np.nanmean(r['ess'])),
        }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {results_dir}/")


if __name__ == '__main__':
    main()
