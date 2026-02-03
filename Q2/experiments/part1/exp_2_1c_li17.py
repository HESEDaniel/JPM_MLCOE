"""Li et al. (2017) skewed-t Poisson experiment."""
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.filters.ekf import ekf_predict, ekf_update
from src.filters.ukf import ukf_predict, ukf_update
from src.filters.pf import systematic_resample
from src.flows.edh import compute_edh_matrices
from src.flows.ledh import compute_ledh_matrices
from src.utils import exponential_lambda_schedule
from src.utils.experiment_logger import ExperimentLogger
from src.ssm.skewed_t_poisson import SkewedTPoissonSSM

@dataclass
class FilterResult:
    """Container for filter results."""
    name: str
    m_filt: np.ndarray
    P_filt: np.ndarray
    runtime: float
    ess: Optional[np.ndarray] = None
    resample_count: int = 0
    mse: Optional[float] = None
    avg_error: Optional[float] = None  # Average estimation error for Lost Track detection

def run_ekf_poisson(model, ys, m0, P0):
    """EKF with state-dependent R for Poisson observations."""
    T, n_x = len(ys), model.d
    m, P = m0.copy(), P0.copy()
    m_filt, P_filt = np.zeros((T, n_x)), np.zeros((T, n_x, n_x))

    start = time.time()
    for t in range(T):
        m_pred, P_pred = ekf_predict(m, P, model.f, model.F_jac, model.Q)
        R_t = model.R_state_dependent(m_pred)
        m, P = ekf_update(m_pred, P_pred, ys[t], model.h, model.H_jac, R_t, joseph=True)
        m_filt[t], P_filt[t] = m, P

    return FilterResult('EKF', m_filt, P_filt, time.time() - start)

def run_ukf_poisson(model, ys, m0, P0):
    """UKF with state-dependent R for Poisson observations."""
    T, n_x = len(ys), model.d
    m, P = m0.copy(), P0.copy()
    m_filt, P_filt = np.zeros((T, n_x)), np.zeros((T, n_x, n_x))

    start = time.time()
    for t in range(T):
        m_pred, P_pred = ukf_predict(m, P, model.f, model.Q, alpha=0.1, kappa=0.0, beta=2.0)
        R_t = model.R_state_dependent(m_pred)
        m, P = ukf_update(m_pred, P_pred, ys[t], model.h, R_t, alpha=0.1, kappa=0.0, beta=2.0, joseph=True)
        m_filt[t], P_filt[t] = m, P

    return FilterResult('UKF', m_filt, P_filt, time.time() - start)

def run_bpf_poisson(model, ys, m0, P0, N, rng, name=None):
    """Bootstrap PF with Poisson likelihood and skewed-t proposal."""
    T, n_x = len(ys), model.d
    particles = rng.multivariate_normal(m0, P0, size=N)
    m_filt, P_filt = np.zeros((T, n_x)), np.zeros((T, n_x, n_x))
    ess_history = np.zeros(T)
    resample_count = 0

    start = time.time()
    for t in range(T):
        for i in range(N):
            particles[i] = model.sample_skewed_t(model.f(particles[i]), rng)

        log_w = model.log_likelihood(ys[t], particles)
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w /= w.sum()

        ess = 1.0 / (w ** 2).sum()
        ess_history[t] = ess

        if ess < 0.5 * N:
            particles = particles[systematic_resample(w, rng)]
            w = np.ones(N) / N
            resample_count += 1

        m_filt[t] = w @ particles
        diff = particles - m_filt[t]
        P_filt[t] = np.einsum('i,ij,ik->jk', w, diff, diff)

    if name is None:
        name = f'BPF({N//1000}K)' if N >= 100000 else f'BPF({N})'

    return FilterResult(name, m_filt, P_filt, time.time() - start, ess_history, resample_count)

def run_edh_poisson(model, ys, m0, P0, N, n_flow_steps, rng):
    """EDH flow with state-dependent R."""
    T, n_x = len(ys), model.d
    lam_sched = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)
    particles = rng.multivariate_normal(m0, P0, size=N)
    m_filt, P_filt = np.zeros((T, n_x)), np.zeros((T, n_x, n_x))
    m, P = m0.copy(), P0.copy()

    start = time.time()
    for t in range(T):
        for i in range(N):
            particles[i] = model.f(particles[i]) + rng.multivariate_normal(np.zeros(n_x), model.Q)

        m_pred, P_pred = ekf_predict(m, P, model.f, model.F_jac, model.Q)
        x, eta_bar = particles.copy(), m_pred.copy()

        for j in range(1, n_flow_steps + 1):
            eps = lam_sched[j] - lam_sched[j - 1]
            R_t = model.R_state_dependent(eta_bar)
            A, b = compute_edh_matrices(m_pred, P_pred, model.H_jac(eta_bar), R_t, ys[t], lam_sched[j], eta_bar, model.h)
            eta_bar = eta_bar + eps * (A @ eta_bar + b)
            x = x + eps * (x @ A.T + b)

        particles = x
        _, P_post = ekf_update(m_pred, P_pred, ys[t], model.h, model.H_jac, model.R_state_dependent(m_pred), joseph=True)
        m_filt[t], P_filt[t] = np.mean(particles, axis=0), P_post
        m, P = m_filt[t].copy(), P_filt[t].copy()

    return FilterResult('EDH', m_filt, P_filt, time.time() - start)

def run_ledh_poisson(model, ys, m0, P0, N, n_flow_steps, rng):
    """LEDH flow with state-dependent R."""
    T, n_x = len(ys), model.d
    lam_sched = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)
    particles = rng.multivariate_normal(m0, P0, size=N)
    m_filt, P_filt = np.zeros((T, n_x)), np.zeros((T, n_x, n_x))
    m, P = m0.copy(), P0.copy()
    I = np.eye(n_x)

    start = time.time()
    for t in range(T):
        for i in range(N):
            particles[i] = model.f(particles[i]) + rng.multivariate_normal(np.zeros(n_x), model.Q)

        m_pred, P_pred = ekf_predict(m, P, model.f, model.F_jac, model.Q)
        x = particles.copy()

        for j in range(1, n_flow_steps + 1):
            eps = lam_sched[j] - lam_sched[j - 1]
            for i in range(N):
                R_i = model.R_state_dependent(x[i])
                R_inv_i = np.diag(1.0 / np.diag(R_i))
                A_i, b_i = compute_ledh_matrices(x[i], m_pred, P_pred, model.h, model.H_jac, R_i, ys[t], lam_sched[j], R_inv_i, I)
                x[i] = x[i] + eps * (A_i @ x[i] + b_i)

        particles = x
        _, P_post = ekf_update(m_pred, P_pred, ys[t], model.h, model.H_jac, model.R_state_dependent(m_pred), joseph=True)
        m_filt[t], P_filt[t] = np.mean(particles, axis=0), P_post
        m, P = m_filt[t].copy(), P_filt[t].copy()

    return FilterResult('LEDH', m_filt, P_filt, time.time() - start)

def _log_gaussian(x, mean, cov_inv, log_det_cov, n):
    diff = x - mean
    return -0.5 * (log_det_cov + n * np.log(2 * np.pi) + diff @ cov_inv @ diff)

def run_pfpf_edh_poisson(model, ys, m0, P0, N, n_flow_steps, rng):
    """PF-PF with EDH flow and state-dependent R."""
    T, n_x = len(ys), model.d
    lam_sched = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)
    Q_inv, log_det_Q = np.linalg.inv(model.Q), np.linalg.slogdet(model.Q)[1]

    particles = rng.multivariate_normal(m0, P0, size=N)
    weights = np.ones(N) / N
    x_hat, P_hat = m0.copy(), P0.copy()
    m_filt, P_filt = np.zeros((T, n_x)), np.zeros((T, n_x, n_x))
    ess_history = np.zeros(T)
    resample_count = 0

    start = time.time()
    for t in range(T):
        m_pred, P_pred = ekf_predict(x_hat, P_hat, model.f, model.F_jac, model.Q)

        eta_0, f_x_prev = np.zeros((N, n_x)), np.zeros((N, n_x))
        for i in range(N):
            f_x_prev[i] = model.f(particles[i])
            eta_0[i] = rng.multivariate_normal(f_x_prev[i], model.Q)

        eta_1 = eta_0.copy()
        eta_bar = model.f(x_hat).copy()

        for j in range(1, n_flow_steps + 1):
            eps = lam_sched[j] - lam_sched[j - 1]
            R_t = model.R_state_dependent(eta_bar)
            A, b = compute_edh_matrices(model.f(x_hat), P_pred, model.H_jac(eta_bar), R_t, ys[t], lam_sched[j], eta_bar, model.h)
            eta_bar = eta_bar + eps * (A @ eta_bar + b)
            eta_1 = eta_1 + eps * (eta_1 @ A.T + b)

        particles = eta_1

        log_lik = model.log_likelihood(ys[t], particles)
        log_trans = np.array([_log_gaussian(particles[i], f_x_prev[i], Q_inv, log_det_Q, n_x) -
                              _log_gaussian(eta_0[i], f_x_prev[i], Q_inv, log_det_Q, n_x) for i in range(N)])

        log_weights = log_lik + log_trans + np.log(weights)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()

        _, P_post = ekf_update(m_pred, P_pred, ys[t], model.h, model.H_jac, model.R_state_dependent(m_pred), joseph=True)
        m_filt[t], P_filt[t] = weights @ particles, P_post
        x_hat, P_hat = m_filt[t].copy(), P_filt[t].copy()

        ess = 1.0 / (weights ** 2).sum()
        ess_history[t] = ess
        if ess < 0.5 * N:
            particles = particles[systematic_resample(weights, rng)]
            weights = np.ones(N) / N
            resample_count += 1

    return FilterResult('PF-PF(EDH)', m_filt, P_filt, time.time() - start, ess_history, resample_count)

def run_pfpf_ledh_poisson(model, ys, m0, P0, N, n_flow_steps, rng):
    """PF-PF with LEDH flow and state-dependent R."""
    T, n_x = len(ys), model.d
    lam_sched = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)
    Q_inv, log_det_Q = np.linalg.inv(model.Q), np.linalg.slogdet(model.Q)[1]
    I = np.eye(n_x)

    particles = rng.multivariate_normal(m0, P0, size=N)
    weights = np.ones(N) / N
    m_prev, P_prev = np.tile(m0, (N, 1)), np.tile(P0, (N, 1, 1))
    m_filt, P_filt = np.zeros((T, n_x)), np.zeros((T, n_x, n_x))
    ess_history = np.zeros(T)
    resample_count = 0

    start = time.time()
    for t in range(T):
        eta_0, eta_1, f_x_prev = np.zeros((N, n_x)), np.zeros((N, n_x)), np.zeros((N, n_x))
        log_theta, P_pred_all, m_pred = np.zeros(N), np.zeros((N, n_x, n_x)), np.zeros((N, n_x))

        for i in range(N):
            m_pred[i], P_pred_all[i] = ekf_predict(particles[i], P_prev[i], model.f, model.F_jac, model.Q)
            f_x_prev[i] = model.f(particles[i])
            eta_0[i] = rng.multivariate_normal(f_x_prev[i], model.Q)
            eta_1[i] = eta_0[i].copy()

        for j in range(1, n_flow_steps + 1):
            eps = lam_sched[j] - lam_sched[j - 1]
            for i in range(N):
                R_i = model.R_state_dependent(eta_1[i])
                R_inv_i = np.diag(1.0 / np.diag(R_i))
                A_i, b_i = compute_ledh_matrices(eta_1[i], f_x_prev[i], P_pred_all[i], model.h, model.H_jac, R_i, ys[t], lam_sched[j], R_inv_i, I)
                eta_1[i] = eta_1[i] + eps * (A_i @ eta_1[i] + b_i)
                log_theta[i] += np.linalg.slogdet(I + eps * A_i)[1]
            log_theta -= np.max(log_theta)

        particles = eta_1

        log_lik = model.log_likelihood(ys[t], particles)
        log_trans = np.array([_log_gaussian(particles[i], f_x_prev[i], Q_inv, log_det_Q, n_x) -
                              _log_gaussian(eta_0[i], f_x_prev[i], Q_inv, log_det_Q, n_x) for i in range(N)])

        log_weights = log_lik + log_theta + log_trans + np.log(weights)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()

        for i in range(N):
            R_i = model.R_state_dependent(m_pred[i])
            m_prev[i], P_prev[i] = ekf_update(m_pred[i], P_pred_all[i], ys[t], model.h, model.H_jac, R_i, joseph=True)

        m_filt[t] = weights @ particles
        diff = particles - m_filt[t]
        P_filt[t] = np.einsum('i,ij,ik->jk', weights, diff, diff)

        ess = 1.0 / (weights ** 2).sum()
        ess_history[t] = ess
        if ess < 0.5 * N:
            idx = systematic_resample(weights, rng)
            particles, m_prev, P_prev = particles[idx], m_prev[idx], P_prev[idx]
            weights = np.ones(N) / N
            resample_count += 1

    return FilterResult('PF-PF(LEDH)', m_filt, P_filt, time.time() - start, ess_history, resample_count)

def compute_mse(xs_true, m_filt):
    """Compute mean squared error."""
    return np.mean((xs_true - m_filt) ** 2)

def compute_avg_error(xs_true, m_filt):
    """Compute average estimation error (MSE) for Lost Track detection."""
    # MSE = mean squared error (averaged over time and dimensions)
    return np.mean((xs_true - m_filt) ** 2)

def run_single_trial(model, rng, N, n_flow_steps, algorithms):
    """Run a single trial of the experiment."""
    xs_true, ys = model.simulate(10, rng, x0=np.zeros(model.d))
    m0, P0 = np.zeros(model.d), model.Q.copy()
    results = {}

    runners = {
        'EKF': lambda: run_ekf_poisson(model, ys, m0, P0),
        'UKF': lambda: run_ukf_poisson(model, ys, m0, P0),
        'EDH': lambda: run_edh_poisson(model, ys, m0, P0, N, n_flow_steps, rng),
        'LEDH': lambda: run_ledh_poisson(model, ys, m0, P0, N, n_flow_steps, rng),
        'PF-PF(EDH)': lambda: run_pfpf_edh_poisson(model, ys, m0, P0, N, n_flow_steps, rng),
        'PF-PF(LEDH)': lambda: run_pfpf_ledh_poisson(model, ys, m0, P0, N, n_flow_steps, rng),
        'PF-PF(EDH)(10K)': lambda: run_pfpf_edh_poisson(model, ys, m0, P0, 10000, n_flow_steps, rng),
        'BPF': lambda: run_bpf_poisson(model, ys, m0, P0, N, rng, name='BPF'),
        'BPF(100K)': lambda: run_bpf_poisson(model, ys, m0, P0, 100000, rng, name='BPF(100K)'),
    }

    for algo in algorithms:
        try:
            if algo in runners:
                results[algo] = runners[algo]()
        except Exception as e:
            print(f"    Warning: {algo} failed: {e}")

    for r in results.values():
        r.mse = compute_mse(xs_true, r.m_filt)
        r.avg_error = compute_avg_error(xs_true, r.m_filt)

    return results

def save_table4(metrics, d_values, save_path, n_trials):
    """Save results to Table 4 format."""
    with open(save_path, 'w') as f:
        f.write('Table IV: Li et al. (2017) Section 5C - Skewed-t with Poisson\n')
        f.write('=' * 80 + '\n')
        for d in d_values:
            d_key = f"d_{d}"
            if d_key not in metrics:
                continue
            f.write(f'\nd = {d} ({int(np.sqrt(d))}x{int(np.sqrt(d))} grid), {n_trials} trials\n')
            f.write(f"{'Algorithm':<15} {'Particles':<10} {'MSE':>12} {'Lost':>8} {'ESS':>10} {'Time':>10}\n")
            f.write('-' * 70 + '\n')
            for algo, m in metrics[d_key].items():
                ess_str = f"{m.get('mean_ess', 0):.1f}" if 'mean_ess' in m else 'N/A'
                lost_tracks = m.get('lost_tracks', 0)
                # Format MSE with lost tracks in brackets if any
                mse_val = m['mean_mse'] if not np.isnan(m['mean_mse']) else m.get('mean_mse_all', np.nan)
                if np.isnan(mse_val):
                    mse_str = "N/A"
                else:
                    mse_str = f"{mse_val:.4f}"
                lost_str = f"({lost_tracks})" if lost_tracks > 0 else "0"
                f.write(f"{algo:<15} {str(m['n_particles']):<10} {mse_str:>12} {lost_str:>8} {ess_str:>10} {m['mean_runtime']:>10.2f}\n")
    print(f'Saved: {save_path}')

def plot_mse_comparison(mse_by_algo, d_values, save_path):
    """Plot MSE comparison across filters."""
    fig, axes = plt.subplots(1, len(d_values), figsize=(6 * len(d_values), 5))
    axes = [axes] if len(d_values) == 1 else axes

    for ax, d in zip(axes, d_values):
        if d in mse_by_algo and mse_by_algo[d]:
            algos = list(mse_by_algo[d].keys())
            ax.bar(algos, [mse_by_algo[d][a] for a in algos], color=plt.cm.tab10(np.linspace(0, 1, len(algos))))
            ax.set(xlabel='Algorithm', ylabel='MSE', title=f'd={d}')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def run_experiment(d_values=None, n_trials=100, N_particles=200, n_flow_steps=29, seed=42,
                   result_dir=None, algorithms=None):
    """Run the experiment."""
    if d_values is None:
        d_values = [144]
    if algorithms is None:
        algorithms = ['EKF', 'UKF', 'EDH', 'LEDH', 'PF-PF(EDH)', 'PF-PF(LEDH)', 'BPF']

    logger = ExperimentLogger(experiment_name='exp_2_1c_li17')

    if result_dir is None:
        result_dir = logger.create_timestamped_run_dir()
    figs_dir, metrics_dir = os.path.join(result_dir, 'figures'), os.path.join(result_dir, 'metrics')
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    pf_algos = ['BPF', 'BPF(100K)', 'PF-PF(EDH)', 'PF-PF(LEDH)', 'PF-PF(EDH)(10K)']
    all_metrics, mse_by_algo = {}, {}

    for d in d_values:
        lost_track_threshold = np.sqrt(d)  # Li et al. (2017): lost if avg_error > sqrt(d)
        model = SkewedTPoissonSSM(d=d, alpha=0.9, alpha0=3.0, alpha1=0.01, beta=20.0, gamma_val=0.3, nu=7.0, m1=1.0, m2=1/3)

        d_key = f"d_{d}"
        all_metrics[d_key], mse_by_algo[d] = {}, {}
        mse_trials = {a: [] for a in algorithms}
        avg_error_trials = {a: [] for a in algorithms}  # For Lost Track detection
        ess_trials = {a: [] for a in algorithms if a in pf_algos}
        time_trials = {a: [] for a in algorithms}

        rng = np.random.default_rng(seed)
        for trial in range(n_trials):
            results = run_single_trial(model, rng, N_particles, n_flow_steps, algorithms)
            for algo, r in results.items():
                mse_trials[algo].append(r.mse)
                avg_error_trials[algo].append(r.avg_error)
                time_trials[algo].append(r.runtime)
                if r.ess is not None and algo in ess_trials:
                    ess_trials[algo].append(np.mean(r.ess))

        for algo in algorithms:
            if not mse_trials[algo]:
                continue

            # Count lost tracks and compute MSE excluding them
            mse_arr = np.array(mse_trials[algo])
            avg_err_arr = np.array(avg_error_trials[algo])
            lost_mask = avg_err_arr > lost_track_threshold
            lost_tracks = np.sum(lost_mask)
            valid_mask = ~lost_mask

            if np.sum(valid_mask) > 0:
                mean_mse_excl = np.mean(mse_arr[valid_mask])
                std_mse_excl = np.std(mse_arr[valid_mask])
            else:
                mean_mse_excl = np.nan
                std_mse_excl = np.nan

            if algo in ['EDH', 'LEDH', 'PF-PF(EDH)', 'PF-PF(LEDH)', 'BPF']:
                n_part = N_particles
            elif algo in ['EKF', 'UKF']:
                n_part = 'N/A'
            elif algo == 'PF-PF(EDH)(10K)':
                n_part = 10000
            else:  # BPF(100K)
                n_part = 100000
            all_metrics[d_key][algo] = {
                'n_particles': n_part,
                'mean_mse': mean_mse_excl,  # MSE excluding lost tracks
                'mean_mse_all': np.mean(mse_arr),  # MSE including all trials
                'std_mse': std_mse_excl,
                'mean_runtime': np.mean(time_trials[algo]),
                'lost_tracks': int(lost_tracks),
                'n_valid': int(np.sum(valid_mask)),
            }
            mse_by_algo[d][algo] = mean_mse_excl
            if algo in ess_trials and ess_trials[algo]:
                all_metrics[d_key][algo]['mean_ess'] = np.mean(ess_trials[algo])

        for algo, m in all_metrics[d_key].items():
            ess_str = f"{m.get('mean_ess', 0):.1f}" if 'mean_ess' in m else 'N/A'
            lost_str = f"({m['lost_tracks']})" if m['lost_tracks'] > 0 else "0"
            mse_val = m["mean_mse"] if not np.isnan(m["mean_mse"]) else m["mean_mse_all"]

    save_table4(all_metrics, d_values, os.path.join(metrics_dir, 'table4.txt'), n_trials)
    plot_mse_comparison(mse_by_algo, d_values, os.path.join(figs_dir, 'mse_comparison.png'))

    return {'metrics': all_metrics, 'mse_by_algo': mse_by_algo}

if __name__ == '__main__':
    # Run only EKF and UKF with Lost Tracks calculation
    run_experiment(d_values=[144], n_trials=100, N_particles=200, n_flow_steps=29,
                   algorithms=['EKF', 'UKF'])
