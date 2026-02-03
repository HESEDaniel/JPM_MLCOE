"""Li et al. (2017) spatial sensor network experiment."""
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.filters.kf import kalman_filter
from src.filters.ukf import unscented_kalman_filter
from src.filters.pf import particle_filter
from src.flows.edh import exact_daum_huang_flow
from src.flows.ledh import local_edh_flow
from src.flows.pfpf_edh import pfpf_edh
from src.flows.pfpf_ledh import pfpf_ledh
from src.utils import exponential_lambda_schedule
from src.utils.experiment_logger import ExperimentLogger
from src.ssm.spatial_sensor_network import SpatialSensorNetwork

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

def run_kf(model, ys, m0, P0):
    """Run Kalman Filter."""
    A, B, C, D = model.get_kf_params()
    start = time.time()
    m_filt, P_filt, _ = kalman_filter(A, B, C, D, P0, ys, joseph=True)
    return FilterResult('KF', m_filt, P_filt, time.time() - start)

def run_ukf(model, ys, m0, P0):
    """Run Unscented Kalman Filter."""
    start = time.time()
    m_filt, P_filt, _ = unscented_kalman_filter(
        model.f, model.h, model.Q, model.R, m0, P0, ys,
        alpha=0.1, kappa=0.0, beta=2.0, joseph=True)
    return FilterResult('UKF', m_filt, P_filt, time.time() - start)

def run_bpf(model, ys, m0, P0, N, rng, name=None):
    """Run Bootstrap Particle Filter."""
    start = time.time()
    m_filt, P_filt, ess, resample_count = particle_filter(
        model.f, model.h, model.Q_sampler, model.log_likelihood,
        m0, P0, ys, N_particles=N, resample_threshold=0.5, rng=rng)
    if name is None:
        name = f'BPF({N//1000}K)' if N >= 100000 else 'BPF'
    return FilterResult(name, m_filt, P_filt, time.time() - start, ess, resample_count)

def run_edh(model, ys, m0, P0, N, n_flow_steps, rng):
    """Run EDH flow filter."""
    T, n_x = len(ys), model.d
    lam_sched = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)
    particles = rng.multivariate_normal(m0, P0, size=N)
    m_filt, P_filt = np.zeros((T, n_x)), np.zeros((T, n_x, n_x))
    m, P = m0.copy(), P0.copy()

    start = time.time()
    for t in range(T):
        particles, _, _, m, P = exact_daum_huang_flow(
            particles, m, P, model.f, model.F_jac, model.Q,
            model.h, model.H_jac, model.R, ys[t],
            lambda_schedule=lam_sched, redraw=True, rng=rng, filter_type='ekf')
        m_filt[t] = np.mean(particles, axis=0)
        diff = particles - m_filt[t]
        P_filt[t] = (diff.T @ diff) / (N - 1)

    return FilterResult('EDH', m_filt, P_filt, time.time() - start)

def run_ledh(model, ys, m0, P0, N, n_flow_steps, rng):
    """Run LEDH flow filter."""
    T, n_x = len(ys), model.d
    lam_sched = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)
    particles = rng.multivariate_normal(m0, P0, size=N)
    m_filt, P_filt = np.zeros((T, n_x)), np.zeros((T, n_x, n_x))
    m, P = m0.copy(), P0.copy()

    start = time.time()
    for t in range(T):
        particles, _, _, m, P = local_edh_flow(
            particles, m, P, model.f, model.F_jac, model.Q,
            model.h, model.H_jac, model.R, ys[t],
            lambda_schedule=lam_sched, store_history=False, redraw=True, rng=rng, filter_type='ekf')
        m_filt[t] = np.mean(particles, axis=0)
        diff = particles - m_filt[t]
        P_filt[t] = (diff.T @ diff) / (N - 1)

    return FilterResult('LEDH', m_filt, P_filt, time.time() - start)

def run_pf_pf_edh(model, ys, m0, P0, N, n_flow_steps, rng, name=None):
    """Run PF-PF with EDH flow."""
    lam_sched = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)
    start = time.time()
    m_filt, P_filt, ess, resample_count, _ = pfpf_edh(
        model.f, model.h, model.H_jac, model.Q, model.R, m0, P0, ys,
        N_particles=N, lambda_schedule=lam_sched, resample_threshold=0.5,
        filter_type='ekf', F_jacobian=model.F_jac, rng=rng)
    if name is None:
        name = 'PF-PF(EDH)'
    return FilterResult(name, m_filt, P_filt, time.time() - start, ess, resample_count)

def run_pf_pf_ledh(model, ys, m0, P0, N, n_flow_steps, rng):
    """Run PF-PF with LEDH flow."""
    lam_sched = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)
    start = time.time()
    m_filt, P_filt, ess, resample_count, _ = pfpf_ledh(
        model.f, model.h, model.H_jac, model.Q, model.R, m0, P0, ys,
        N_particles=N, lambda_schedule=lam_sched, resample_threshold=0.5,
        filter_type='ekf', F_jacobian=model.F_jac, rng=rng)
    return FilterResult('PF-PF(LEDH)', m_filt, P_filt, time.time() - start, ess, resample_count)

def compute_mse(xs_true, m_filt):
    """Compute mean squared error."""
    return np.mean((xs_true - m_filt) ** 2)

def run_single_trial(model, rng, N, n_flow_steps, algorithms):
    """Run a single trial of the experiment."""
    xs_true, ys = model.simulate(10, rng, x0=np.zeros(model.d))
    m0, P0 = np.zeros(model.d), model.Q.copy()
    results = {}

    runners = {
        'KF': lambda: run_kf(model, ys, m0, P0),
        'UKF': lambda: run_ukf(model, ys, m0, P0),
        'BPF': lambda: run_bpf(model, ys, m0, P0, N, rng, name='BPF'),
        'BPF(100K)': lambda: run_bpf(model, ys, m0, P0, 100000, rng, name='BPF(100K)'),
        'EDH': lambda: run_edh(model, ys, m0, P0, N, n_flow_steps, rng),
        'LEDH': lambda: run_ledh(model, ys, m0, P0, N, n_flow_steps, rng),
        'PF-PF(EDH)': lambda: run_pf_pf_edh(model, ys, m0, P0, N, n_flow_steps, rng),
        'PF-PF(EDH)(10K)': lambda: run_pf_pf_edh(model, ys, m0, P0, 10000, n_flow_steps, rng, name='PF-PF(EDH)(10K)'),
        'PF-PF(LEDH)': lambda: run_pf_pf_ledh(model, ys, m0, P0, N, n_flow_steps, rng),
    }

    for algo in algorithms:
        try:
            if algo in runners:
                results[algo] = runners[algo]()
        except Exception as e:
            print(f"    Warning: {algo} failed: {e}")

    for r in results.values():
        r.mse = compute_mse(xs_true, r.m_filt)

    return results

def save_table2(metrics, sigma_z_values, save_path):
    """Save results to Table 2 format."""
    with open(save_path, 'w') as f:
        f.write('Table II: Li et al. (2017) Section 5B - Spatial Sensor Networks\n')
        f.write('=' * 80 + '\n')
        f.write(f"{'Algorithm':<15} {'Part.':<8}")
        for sigma in sigma_z_values:
            f.write(f"{'sigma=' + str(sigma):^16}")
        f.write(f"{'Time':>10}\n")
        f.write('-' * 80 + '\n')

        for algo, m in metrics.items():
            n_part = str(m.get('n_particles', 'N/A'))
            f.write(f"{algo:<15} {n_part:<8}")
            for sigma in sigma_z_values:
                key = f"sigma_{sigma}"
                if key in m:
                    mse = m[key].get('mean_mse', float('nan'))
                    ess = m[key].get('mean_ess', None)
                    ess_str = f"{ess:.0f}" if ess else "N/A"
                    f.write(f"{mse:>8.3f} {ess_str:>6}")
                else:
                    f.write(f"{'N/A':>8} {'N/A':>6}")
            f.write(f"{m.get('mean_runtime', 0):>10.2f}\n")
    print(f'Saved: {save_path}')

def plot_mse_comparison(mse_by_algo, sigma_z_values, save_path):
    """Plot MSE comparison across filters."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sigma_z_values))
    n_algos = len(mse_by_algo)
    width = 0.8 / n_algos
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_algos)

    for i, (algo, mse_dict) in enumerate(mse_by_algo.items()):
        vals = [mse_dict.get(s, float('nan')) for s in sigma_z_values]
        ax.bar(x + offsets[i], vals, width, label=algo)

    ax.set(xlabel='sigma_z', ylabel='MSE', title='MSE Comparison (Li17 Section 5B)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sigma_z_values])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def run_experiment(d=64, n_trials=100, sigma_z_values=None, N_particles=200,
                   n_flow_steps=29, seed=42, result_dir=None, algorithms=None):
    """Run the experiment."""
    if sigma_z_values is None:
        sigma_z_values = [2.0, 1.0, 0.5]
    if algorithms is None:
        algorithms = ['KF', 'UKF', 'EDH', 'PF-PF(EDH)', 'PF-PF(LEDH)', 'BPF']

    logger = ExperimentLogger(experiment_name='exp_2_1b_li17')

    if result_dir is None:
        result_dir = logger.create_timestamped_run_dir()
    figs_dir, metrics_dir = os.path.join(result_dir, 'figures'), os.path.join(result_dir, 'metrics')
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    pf_algos = ['BPF', 'BPF(100K)', 'PF-PF(EDH)', 'PF-PF(EDH)(10K)', 'PF-PF(LEDH)']

    def get_n_particles(algo):
        """Get number of particles for each filter."""
        if algo in ['EDH', 'LEDH', 'PF-PF(EDH)', 'PF-PF(LEDH)', 'BPF']:
            return N_particles
        elif algo == 'BPF(100K)':
            return 100000
        elif algo == 'PF-PF(EDH)(10K)':
            return 10000
        else:
            return 'N/A'

    all_metrics = {a: {'n_particles': get_n_particles(a), 'runtimes': []} for a in algorithms}
    mse_by_algo = {a: {} for a in algorithms}

    for sigma_z in sigma_z_values:
        model = SpatialSensorNetwork(d=d, sigma_z=sigma_z)

        mse_trials = {a: [] for a in algorithms}
        ess_trials = {a: [] for a in algorithms if a in pf_algos}
        time_trials = {a: [] for a in algorithms}

        rng = np.random.default_rng(seed)
        for trial in range(n_trials):
            results = run_single_trial(model, rng, N_particles, n_flow_steps, algorithms)
            for algo, r in results.items():
                mse_trials[algo].append(r.mse)
                time_trials[algo].append(r.runtime)
                if r.ess is not None and algo in ess_trials:
                    ess_trials[algo].append(np.mean(r.ess))

        sigma_key = f"sigma_{sigma_z}"
        for algo in algorithms:
            if not mse_trials[algo]:
                continue
            all_metrics[algo][sigma_key] = {
                'mean_mse': np.mean(mse_trials[algo]),
                'std_mse': np.std(mse_trials[algo]),
            }
            all_metrics[algo]['runtimes'].extend(time_trials[algo])
            mse_by_algo[algo][sigma_z] = np.mean(mse_trials[algo])
            if algo in ess_trials and ess_trials[algo]:
                all_metrics[algo][sigma_key]['mean_ess'] = np.mean(ess_trials[algo])

        for algo in algorithms:
            if sigma_key in all_metrics[algo]:
                m = all_metrics[algo][sigma_key]
                ess_str = f"{m.get('mean_ess', 0):.1f}" if 'mean_ess' in m else 'N/A'

    for algo in algorithms:
        if all_metrics[algo]['runtimes']:
            all_metrics[algo]['mean_runtime'] = np.mean(all_metrics[algo]['runtimes'])

    save_table2(all_metrics, sigma_z_values, os.path.join(metrics_dir, 'table2.txt'))
    plot_mse_comparison(mse_by_algo, sigma_z_values, os.path.join(figs_dir, 'mse_comparison.png'))

    return {'metrics': all_metrics, 'mse_by_algo': mse_by_algo}

if __name__ == '__main__':
    run_experiment(d=64, n_trials=10, sigma_z_values=[2.0, 1.0, 0.5], N_particles=200,
                   n_flow_steps=29, algorithms=['PF-PF(LEDH)', 'PF-PF(EDH)', 'PF-PF(EDH)(10K)', 'EDH', 'KF', 'UKF', 'BPF', 'BPF(100K)'])
