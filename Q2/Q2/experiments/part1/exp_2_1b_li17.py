"""Li et al. (2017) spatial sensor network experiment (TensorFlow)."""
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

from src.ssm.spatial_sensor_network import SpatialSensorNetwork
from src.filters import KalmanFilter, UnscentedKalmanFilter, ParticleFilter
from src.flows import EDHFlow, LEDHFlow, PFPFEDHFilter, PFPFLEDHFilter

DTYPE = tf.float64


def exponential_lambda_schedule(n_steps, ratio=1.2):
    raw = np.array([ratio**i for i in range(n_steps)])
    cumsum = np.cumsum(raw)
    return cumsum / cumsum[-1]


@dataclass
class FResult:
    name: str
    m_filt: np.ndarray
    runtime: float
    ess: Optional[np.ndarray] = None
    resample_count: int = 0
    mse: Optional[float] = None


def compute_mse(xs_true, m_filt):
    return float(np.mean((xs_true - m_filt) ** 2))


def run_single_trial(model, rng_np, N, n_flow_steps, algorithms):
    """Run a single trial."""
    xs_true, ys = model.simulate(10, rng_np, x0=np.zeros(model.state_dim))
    ys_tf = tf.constant(ys, dtype=DTYPE)

    m0_np = np.zeros(model.state_dim)
    P0_np = model.Q.numpy().copy()
    model.m0 = tf.constant(m0_np, dtype=DTYPE)
    model.P0 = tf.constant(P0_np, dtype=DTYPE)

    lam_sched = exponential_lambda_schedule(n_flow_steps, ratio=1.2)
    results = {}

    for algo in algorithms:
        try:
            rng_tf = tf.random.Generator.from_seed(int(rng_np.integers(1e9)))

            if algo == 'KF':
                kf = KalmanFilter(joseph=True)
                t0 = time.perf_counter()
                res = kf.filter(model, ys_tf)
                rt = time.perf_counter() - t0
                results[algo] = FResult(algo, res.m_filt.numpy(), rt)

            elif algo == 'UKF':
                ukf = UnscentedKalmanFilter(alpha=0.1, kappa=0.0, beta=2.0)
                t0 = time.perf_counter()
                res = ukf.filter(model, ys_tf)
                rt = time.perf_counter() - t0
                results[algo] = FResult(algo, res.m_filt.numpy(), rt)

            elif algo.startswith('BPF'):
                n_p = 100000 if '100K' in algo else N
                pf = ParticleFilter(n_particles=n_p, resample_threshold=0.5)
                t0 = time.perf_counter()
                res = pf.filter(model, ys_tf, rng=rng_tf)
                rt = time.perf_counter() - t0
                ess = res.diagnostics.get('ess')
                ess_np = ess.numpy() if ess is not None else None
                results[algo] = FResult(algo, res.m_filt.numpy(), rt, ess_np,
                                        res.diagnostics.get('resample_count', 0))

            elif algo == 'EDH':
                edh = EDHFlow(n_particles=N, n_flow_steps=n_flow_steps,
                              lambda_schedule=lam_sched, filter_type='ekf')
                t0 = time.perf_counter()
                res = edh.filter(model, ys_tf, rng=rng_tf)
                rt = time.perf_counter() - t0
                results[algo] = FResult(algo, res.m_filt.numpy(), rt)

            elif algo == 'LEDH':
                ledh = LEDHFlow(n_particles=N, n_flow_steps=n_flow_steps,
                                lambda_schedule=lam_sched, filter_type='ekf')
                t0 = time.perf_counter()
                res = ledh.filter(model, ys_tf, rng=rng_tf)
                rt = time.perf_counter() - t0
                results[algo] = FResult(algo, res.m_filt.numpy(), rt)

            elif 'PF-PF(EDH)' in algo:
                n_p = 10000 if '10K' in algo else N
                pfpf = PFPFEDHFilter(n_particles=n_p, n_flow_steps=n_flow_steps,
                                    lambda_schedule=lam_sched, filter_type='ekf')
                t0 = time.perf_counter()
                res = pfpf.filter(model, ys_tf, rng=rng_tf)
                rt = time.perf_counter() - t0
                ess = res.diagnostics.get('ess')
                ess_np = ess.numpy() if ess is not None else None
                results[algo] = FResult(algo, res.m_filt.numpy(), rt, ess_np,
                                        res.diagnostics.get('resample_count', 0))

            elif algo == 'PF-PF(LEDH)':
                pfpf = PFPFLEDHFilter(n_particles=N, n_flow_steps=n_flow_steps,
                                     lambda_schedule=lam_sched, filter_type='ekf')
                t0 = time.perf_counter()
                res = pfpf.filter(model, ys_tf, rng=rng_tf)
                rt = time.perf_counter() - t0
                ess = res.diagnostics.get('ess')
                ess_np = ess.numpy() if ess is not None else None
                results[algo] = FResult(algo, res.m_filt.numpy(), rt, ess_np,
                                        res.diagnostics.get('resample_count', 0))

        except Exception as e:
            print(f"    Warning: {algo} failed: {e}")

    for r in results.values():
        r.mse = compute_mse(xs_true, r.m_filt)

    return results


def save_table2(metrics, sigma_z_values, save_path):
    with open(save_path, 'w') as f:
        f.write('Table II: Li et al. (2017) Section 5B - Spatial Sensor Networks (TF)\n')
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
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sigma_z_values))
    n_algos = len(mse_by_algo)
    width = 0.8 / max(n_algos, 1)
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_algos)

    for i, (algo, mse_dict) in enumerate(mse_by_algo.items()):
        vals = [mse_dict.get(s, float('nan')) for s in sigma_z_values]
        ax.bar(x + offsets[i], vals, width, label=algo)

    ax.set(xlabel='sigma_z', ylabel='MSE', title='MSE Comparison (Li17 5B, TF)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sigma_z_values])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_experiment(d=64, n_trials=100, sigma_z_values=None, N_particles=200,
                   n_flow_steps=29, seed=42, result_dir=None, algorithms=None):
    if sigma_z_values is None:
        sigma_z_values = [2.0, 1.0, 0.5]
    if algorithms is None:
        algorithms = ['KF', 'UKF', 'EDH', 'PF-PF(EDH)', 'PF-PF(LEDH)', 'BPF']

    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part1', 'exp_2_1b_li17')
    figs_dir = os.path.join(result_dir, 'figures')
    metrics_dir = os.path.join(result_dir, 'metrics')
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    pf_algos = ['BPF', 'BPF(100K)', 'PF-PF(EDH)', 'PF-PF(EDH)(10K)', 'PF-PF(LEDH)']

    def get_n_particles(algo):
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

    total_t0 = time.perf_counter()

    for sigma_z in sigma_z_values:
        print(f"\n--- sigma_z = {sigma_z} ---")
        model = SpatialSensorNetwork(d=d, sigma_z=sigma_z)

        mse_trials = {a: [] for a in algorithms}
        ess_trials = {a: [] for a in algorithms if a in pf_algos}
        time_trials = {a: [] for a in algorithms}

        rng_np = np.random.default_rng(seed)
        for trial in range(n_trials):
            if (trial + 1) % max(1, n_trials // 5) == 0:
                print(f"  Trial {trial + 1}/{n_trials}")
            results = run_single_trial(model, rng_np, N_particles, n_flow_steps, algorithms)
            for algo, r in results.items():
                mse_trials[algo].append(r.mse)
                time_trials[algo].append(r.runtime)
                if r.ess is not None and algo in ess_trials:
                    ess_trials[algo].append(float(np.mean(r.ess)))

        sigma_key = f"sigma_{sigma_z}"
        for algo in algorithms:
            if not mse_trials[algo]:
                continue
            all_metrics[algo][sigma_key] = {
                'mean_mse': float(np.mean(mse_trials[algo])),
                'std_mse': float(np.std(mse_trials[algo])),
            }
            all_metrics[algo]['runtimes'].extend(time_trials[algo])
            mse_by_algo[algo][sigma_z] = float(np.mean(mse_trials[algo]))
            if algo in ess_trials and ess_trials[algo]:
                all_metrics[algo][sigma_key]['mean_ess'] = float(np.mean(ess_trials[algo]))

    for algo in algorithms:
        if all_metrics[algo]['runtimes']:
            all_metrics[algo]['mean_runtime'] = float(np.mean(all_metrics[algo]['runtimes']))

    save_table2(all_metrics, sigma_z_values, os.path.join(metrics_dir, 'table2.txt'))
    plot_mse_comparison(mse_by_algo, sigma_z_values, os.path.join(figs_dir, 'mse_comparison.png'))

    total_time = time.perf_counter() - total_t0
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved to: {result_dir}")

    return {'metrics': all_metrics, 'mse_by_algo': mse_by_algo}


if __name__ == '__main__':
    run_experiment(d=64, n_trials=10, sigma_z_values=[2.0, 1.0, 0.5], N_particles=200,
                   n_flow_steps=29, algorithms=['PF-PF(LEDH)', 'PF-PF(EDH)', 'EDH', 'KF', 'UKF', 'BPF'])
