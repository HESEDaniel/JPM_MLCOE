"""Li et al. (2017) skewed-t Poisson experiment (TensorFlow).

Note: This experiment uses custom loops with state-dependent R,
so it cannot use the standard filter.filter() API directly.
"""
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm.skewed_t_poisson import SkewedTPoissonSSM
from src.filters import ExtendedKalmanFilter, UnscentedKalmanFilter
from src.filters.common import DTYPE
from src.flows.edh import compute_edh_matrices
from src.flows.ledh import compute_ledh_matrices

def exponential_lambda_schedule(n_steps, ratio=1.2):
    raw = np.array([ratio**i for i in range(n_steps)])
    cumsum = np.cumsum(raw)
    lam = np.concatenate([[0.0], cumsum / cumsum[-1]])
    return lam


@dataclass
class FResult:
    name: str
    m_filt: np.ndarray
    runtime: float
    ess: Optional[np.ndarray] = None
    resample_count: int = 0
    mse: Optional[float] = None
    avg_error: Optional[float] = None


def run_ekf_poisson(model, ys_tf, m0_tf, P0_tf):
    """EKF with state-dependent R."""
    T = ys_tf.shape[0]
    n_x = model.state_dim
    ekf = ExtendedKalmanFilter(joseph=True)
    m, P = m0_tf, P0_tf
    m_filt = np.zeros((T, n_x))

    t0 = time.perf_counter()
    for t in range(T):
        m_pred, P_pred = ekf.predict(m, P, model)
        R_t = model.R_state_dependent(m_pred)
        old_R = model.R
        model.R = R_t
        m, P = ekf.update(m_pred, P_pred, ys_tf[t], model)
        model.R = old_R
        m_filt[t] = m.numpy()

    return FResult('EKF', m_filt, time.perf_counter() - t0)


def run_ukf_poisson(model, ys_tf, m0_tf, P0_tf):
    """UKF with state-dependent R."""
    T = ys_tf.shape[0]
    n_x = model.state_dim
    ukf = UnscentedKalmanFilter(alpha=0.1, kappa=0.0, beta=2.0)
    m, P = m0_tf, P0_tf
    m_filt = np.zeros((T, n_x))

    t0 = time.perf_counter()
    for t in range(T):
        m_pred, P_pred = ukf.predict(m, P, model)
        R_t = model.R_state_dependent(m_pred)
        old_R = model.R
        model.R = R_t
        m, P = ukf.update(m_pred, P_pred, ys_tf[t], model)
        model.R = old_R
        m_filt[t] = m.numpy()

    return FResult('UKF', m_filt, time.perf_counter() - t0)


def run_bpf_poisson(model, ys, m0_np, P0_np, N, rng_np, name=None):
    """Bootstrap PF with skewed-t proposal (numpy for skewed-t sampling)."""
    T, n_x = len(ys), model.state_dim
    particles = rng_np.multivariate_normal(m0_np, P0_np, size=N)
    m_filt = np.zeros((T, n_x))
    ess_history = np.zeros(T)
    resample_count = 0

    t0 = time.perf_counter()
    for t in range(T):
        for i in range(N):
            particles[i] = model.sample_skewed_t(model.f(tf.constant(particles[i], dtype=DTYPE)).numpy(), rng_np)

        particles_tf = tf.constant(particles, dtype=DTYPE)
        y_tf = tf.constant(ys[t], dtype=DTYPE)
        log_w = model.log_likelihood(y_tf, particles_tf).numpy()
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w /= w.sum()

        ess = 1.0 / (w ** 2).sum()
        ess_history[t] = ess

        if ess < 0.5 * N:
            idx = np.searchsorted(np.cumsum(w), rng_np.random(N))
            idx = np.clip(idx, 0, N - 1)
            particles = particles[idx]
            w = np.ones(N) / N
            resample_count += 1

        m_filt[t] = w @ particles

    if name is None:
        name = f'BPF({N//1000}K)' if N >= 100000 else f'BPF({N})'
    return FResult(name, m_filt, time.perf_counter() - t0, ess_history, resample_count)


def run_edh_poisson(model, ys_tf, m0_tf, P0_tf, N, n_flow_steps, rng_np):
    """EDH flow with state-dependent R."""
    T = ys_tf.shape[0]
    n_x = model.state_dim
    lam_sched = exponential_lambda_schedule(n_flow_steps)
    ekf = ExtendedKalmanFilter(joseph=True)

    m0_np, P0_np = m0_tf.numpy(), P0_tf.numpy()
    particles = rng_np.multivariate_normal(m0_np, P0_np, size=N)
    m, P = m0_tf, P0_tf
    m_filt = np.zeros((T, n_x))

    t0 = time.perf_counter()
    for t in range(T):
        # Propagate
        Q_np = model.Q.numpy()
        for i in range(N):
            f_val = model.f(tf.constant(particles[i], dtype=DTYPE)).numpy()
            particles[i] = f_val + rng_np.multivariate_normal(np.zeros(n_x), Q_np)

        m_pred, P_pred = ekf.predict(m, P, model)
        x = tf.constant(particles, dtype=DTYPE)
        eta_bar = m_pred

        # Flow
        for j in range(1, n_flow_steps + 1):
            eps = lam_sched[j] - lam_sched[j - 1]
            R_t = model.R_state_dependent(eta_bar)
            H_curr = model.H_jac(eta_bar)
            A, b = compute_edh_matrices(m_pred, P_pred, H_curr, R_t, ys_tf[t],
                                         tf.constant(lam_sched[j], dtype=DTYPE), eta_bar, model.h)
            eta_bar = eta_bar + eps * (tf.linalg.matvec(A, eta_bar) + b)
            x = x + eps * (x @ tf.transpose(A) + b[tf.newaxis, :])

        particles = x.numpy()
        R_t = model.R_state_dependent(m_pred)
        old_R = model.R
        model.R = R_t
        _, P_post = ekf.update(m_pred, P_pred, ys_tf[t], model)
        model.R = old_R
        m_filt[t] = np.mean(particles, axis=0)
        m, P = tf.constant(m_filt[t], dtype=DTYPE), P_post

    return FResult('EDH', m_filt, time.perf_counter() - t0)


def compute_mse(xs_true, m_filt):
    return float(np.mean((xs_true - m_filt) ** 2))


def run_single_trial(model, rng_np, N, n_flow_steps, algorithms):
    xs_true, ys = model.simulate(10, rng_np, x0=np.zeros(model.state_dim))
    ys_tf = tf.constant(ys, dtype=DTYPE)

    m0_np = np.zeros(model.state_dim)
    P0_np = model.Q.numpy().copy()
    m0_tf = tf.constant(m0_np, dtype=DTYPE)
    P0_tf = tf.constant(P0_np, dtype=DTYPE)

    results = {}
    runners = {
        'EKF': lambda: run_ekf_poisson(model, ys_tf, m0_tf, P0_tf),
        'UKF': lambda: run_ukf_poisson(model, ys_tf, m0_tf, P0_tf),
        'EDH': lambda: run_edh_poisson(model, ys_tf, m0_tf, P0_tf, N, n_flow_steps, rng_np),
        'BPF': lambda: run_bpf_poisson(model, ys, m0_np, P0_np, N, rng_np, name='BPF'),
    }

    for algo in algorithms:
        try:
            if algo in runners:
                results[algo] = runners[algo]()
        except Exception as e:
            print(f"    Warning: {algo} failed: {e}")

    for r in results.values():
        r.mse = compute_mse(xs_true, r.m_filt)
        r.avg_error = r.mse

    return results


def save_table4(metrics, d_values, save_path, n_trials):
    with open(save_path, 'w') as f:
        f.write('Table IV: Li et al. (2017) Section 5C - Skewed-t Poisson (TF)\n')
        f.write('=' * 80 + '\n')
        for d in d_values:
            d_key = f"d_{d}"
            if d_key not in metrics:
                continue
            f.write(f'\nd = {d}, {n_trials} trials\n')
            f.write(f"{'Algorithm':<15} {'MSE':>12} {'Lost':>8} {'Time':>10}\n")
            f.write('-' * 50 + '\n')
            for algo, m in metrics[d_key].items():
                mse_val = m['mean_mse']
                mse_str = f"{mse_val:.4f}" if not np.isnan(mse_val) else "N/A"
                lost_str = f"({m.get('lost_tracks', 0)})"
                f.write(f"{algo:<15} {mse_str:>12} {lost_str:>8} {m['mean_runtime']:>10.2f}\n")
    print(f'Saved: {save_path}')


def run_experiment(d_values=None, n_trials=100, N_particles=200, n_flow_steps=29,
                   seed=42, result_dir=None, algorithms=None):
    if d_values is None:
        d_values = [144]
    if algorithms is None:
        algorithms = ['EKF', 'UKF', 'EDH', 'BPF']

    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part1', 'exp_2_1c_li17')
    figs_dir = os.path.join(result_dir, 'figures')
    metrics_dir = os.path.join(result_dir, 'metrics')
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    all_metrics = {}
    total_t0 = time.perf_counter()

    for d in d_values:
        lost_track_threshold = np.sqrt(d)
        model = SkewedTPoissonSSM(d=d, alpha=0.9, alpha0=3.0, alpha1=0.01,
                                   beta=20.0, gamma_val=0.3, nu=7.0, m1=1.0, m2=1/3)
        d_key = f"d_{d}"
        mse_trials = {a: [] for a in algorithms}
        time_trials = {a: [] for a in algorithms}

        rng_np = np.random.default_rng(seed)
        for trial in range(n_trials):
            if (trial + 1) % max(1, n_trials // 5) == 0:
                print(f"  d={d}, trial {trial + 1}/{n_trials}")
            results = run_single_trial(model, rng_np, N_particles, n_flow_steps, algorithms)
            for algo, r in results.items():
                mse_trials[algo].append(r.mse)
                time_trials[algo].append(r.runtime)

        all_metrics[d_key] = {}
        for algo in algorithms:
            if not mse_trials[algo]:
                continue
            mse_arr = np.array(mse_trials[algo])
            lost_mask = mse_arr > lost_track_threshold
            valid_mask = ~lost_mask
            mean_mse = float(np.mean(mse_arr[valid_mask])) if np.any(valid_mask) else float('nan')

            all_metrics[d_key][algo] = {
                'mean_mse': mean_mse,
                'mean_runtime': float(np.mean(time_trials[algo])),
                'lost_tracks': int(np.sum(lost_mask)),
            }

    save_table4(all_metrics, d_values, os.path.join(metrics_dir, 'table4.txt'), n_trials)

    total_time = time.perf_counter() - total_t0
    print(f"\nTotal time: {total_time:.1f}s")
    return all_metrics


if __name__ == '__main__':
    run_experiment(d_values=[144], n_trials=10, N_particles=200, n_flow_steps=29,
                   algorithms=['EKF', 'UKF'])
