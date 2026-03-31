"""Stochastic Volatility model filter comparison (TensorFlow)."""
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DTYPE = tf.float64

from src.ssm import SVLogTransformed, SVAdditiveNoise
from src.filters import ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter


@dataclass
class FilterMetrics:
    """Container for filter evaluation metrics."""
    name: str
    mse: float
    nees_mean: float
    nees_std: float
    ess_mean: Optional[float]  # Only for PF
    ess_min: Optional[float]   # Only for PF
    runtime_s: float
    m_filt: np.ndarray = field(repr=False)
    P_filt: np.ndarray = field(repr=False)


def compute_nees(m_filt, P_filt, xs_col):
    """Compute Normalised Estimation Error Squared (inline)."""
    T = m_filt.shape[0]
    nees = np.zeros(T)
    for t in range(T):
        diff = (m_filt[t] - xs_col[t]).reshape(-1, 1)
        try:
            nees[t] = float(diff.T @ np.linalg.solve(P_filt[t], diff))
        except np.linalg.LinAlgError:
            nees[t] = np.nan
    return nees


def run_filters(model, xs, ys, N_particles=1000, rng_seed=42) -> Dict[str, FilterMetrics]:
    """Run EKF, UKF, PF on given model and observations with timing."""
    zs = model.transform_obs(ys)
    ys_tf = tf.constant(zs, dtype=DTYPE)
    xs_col = xs.reshape(-1, 1) if xs.ndim == 1 else xs
    results = {}

    # EKF
    ekf = ExtendedKalmanFilter()
    t0 = time.perf_counter()
    ekf_result = ekf.filter(model, ys_tf)
    runtime_ekf = time.perf_counter() - t0

    m_ekf = ekf_result.m_filt.numpy()
    P_ekf = ekf_result.P_filt.numpy()
    nees_ekf = compute_nees(m_ekf, P_ekf, xs_col)
    results['EKF'] = FilterMetrics(
        name='EKF',
        mse=np.mean((m_ekf[:, 0] - xs) ** 2),
        nees_mean=np.nanmean(nees_ekf),
        nees_std=np.nanstd(nees_ekf),
        ess_mean=None,
        ess_min=None,
        runtime_s=runtime_ekf,
        m_filt=m_ekf,
        P_filt=P_ekf,
    )

    # UKF
    ukf = UnscentedKalmanFilter()
    t0 = time.perf_counter()
    ukf_result = ukf.filter(model, ys_tf)
    runtime_ukf = time.perf_counter() - t0

    m_ukf = ukf_result.m_filt.numpy()
    P_ukf = ukf_result.P_filt.numpy()
    nees_ukf = compute_nees(m_ukf, P_ukf, xs_col)
    results['UKF'] = FilterMetrics(
        name='UKF',
        mse=np.mean((m_ukf[:, 0] - xs) ** 2),
        nees_mean=np.nanmean(nees_ukf),
        nees_std=np.nanstd(nees_ukf),
        ess_mean=None,
        ess_min=None,
        runtime_s=runtime_ukf,
        m_filt=m_ukf,
        P_filt=P_ukf,
    )

    # PF
    pf = ParticleFilter(n_particles=N_particles)
    rng_tf = tf.random.Generator.from_seed(rng_seed)

    # PF uses raw observations (not log-transformed) for SV models
    ys_pf = tf.constant(ys.reshape(-1, 1), dtype=DTYPE)

    t0 = time.perf_counter()
    pf_result = pf.filter(model, ys_pf, rng=rng_tf)
    runtime_pf = time.perf_counter() - t0

    m_pf = pf_result.m_filt.numpy()
    P_pf = pf_result.P_filt.numpy()
    ess_pf = pf_result.diagnostics['ess'].numpy()
    nees_pf = compute_nees(m_pf, P_pf, xs_col)
    results['PF'] = FilterMetrics(
        name='PF',
        mse=np.mean((m_pf[:, 0] - xs) ** 2),
        nees_mean=np.nanmean(nees_pf),
        nees_std=np.nanstd(nees_pf),
        ess_mean=np.mean(ess_pf),
        ess_min=np.min(ess_pf),
        runtime_s=runtime_pf,
        m_filt=m_pf,
        P_filt=P_pf,
    )

    return results


def write_metrics_table(all_results: Dict[str, Dict[str, FilterMetrics]],
                        save_path: str, N_particles: int, T: int):
    """Write evaluation metrics to a formatted text file."""
    with open(save_path, 'w') as f:
        f.write("Stochastic Volatility Filter Comparison (TensorFlow)\n")
        f.write(f"Configuration: T={T}, N_particles={N_particles}\n\n")

        header = (f"{'Model':<25} {'Filter':<6} {'MSE':>10} {'NEES':>12} "
                  f"{'ESS_mean':>10} {'ESS_min':>10} {'Runtime':>10}\n")
        units = (f"{'':25} {'':6} {'':>10} {'(mean+/-std)':>12} "
                 f"{'':>10} {'':>10} {'(s)':>10}\n")

        f.write(header)
        f.write(units)

        for model_name, results in all_results.items():
            for filter_name, m in results.items():
                ess_mean_str = f"{m.ess_mean:.1f}" if m.ess_mean is not None else "N/A"
                ess_min_str = f"{m.ess_min:.1f}" if m.ess_min is not None else "N/A"
                nees_str = f"{m.nees_mean:.2f}+/-{m.nees_std:.2f}"

                f.write(f"{model_name:<25} {filter_name:<6} {m.mse:>10.4f} {nees_str:>12} "
                        f"{ess_mean_str:>10} {ess_min_str:>10} {m.runtime_s:>10.4f}\n")
            f.write("\n")


def plot_data_combined(t, data_list, save_path):
    """Plot true state and observations for multiple models in one figure.

    data_list: list of (xs, ys, title) or (xs, ys, title, log_scale) tuples
    """
    n = len(data_list)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, data_list):
        xs, ys, title = item[0], item[1], item[2]
        log_scale = item[3] if len(item) > 3 else False

        ax.plot(t, xs, 'b-', lw=2, label='True State $x_t$')
        ax.scatter(t, ys, s=10, c='red', alpha=0.5, label='Observations $y_t$')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale('symlog', linthresh=1.0)
            ax.set_ylabel('Value (symlog scale)')

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_filtering_results(t, xs, results: Dict[str, FilterMetrics], title, save_path):
    """Plot filtering results comparing true state vs estimates."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, (name, m) in zip(axes, results.items()):
        std = np.sqrt(m.P_filt[:, 0, 0])

        ax.plot(t, xs, 'k-', label='True State')
        ax.plot(t, m.m_filt[:, 0], 'r-', label=f'{name} Estimate')
        ax.fill_between(t, m.m_filt[:, 0] - std, m.m_filt[:, 0] + std,
                        alpha=0.2, color='red', label='+/- 1 S.D.')

        ax.set_ylabel('State x')
        ax.set_title(f'{name} (MSE={m.mse:.4f}, NEES={m.nees_mean:.2f})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment(rng_np, result_path, T=200, N_particles=1000,
                   alpha=0.91, sigma=1.0, beta=0.5, obs_std=0.5):
    """Run filter comparison experiment."""

    t = np.arange(T)

    # Create models
    model_log = SVLogTransformed(alpha, sigma, beta)
    model_add = SVAdditiveNoise(alpha, sigma, beta, obs_std, exp_scale=0.5)
    model_strong = SVAdditiveNoise(alpha, sigma, beta, obs_std, exp_scale=2.0)

    # Generate data
    xs_log, ys_log = model_log.simulate(T, rng_np)
    xs_add, ys_add = model_add.simulate(T, rng_np)
    xs_strong, ys_strong = model_strong.simulate(T, rng_np)

    # Run filters with timing
    print("  Running filters on Log-Transformed model...")
    results_log = run_filters(model_log, xs_log, ys_log, N_particles, rng_seed=42)

    print("  Running filters on Additive Noise model...")
    results_add = run_filters(model_add, xs_add, ys_add, N_particles, rng_seed=43)

    print("  Running filters on Strong Nonlinear model...")
    results_strong = run_filters(model_strong, xs_strong, ys_strong, N_particles, rng_seed=44)

    # Collect all results for table output
    all_results = {
        'Log-Transformed (Linear)': results_log,
        'Additive (Nonlinear)': results_add,
        'Additive (Strong Nonlinear)': results_strong,
    }

    # Write metrics table
    write_metrics_table(all_results, os.path.join(result_path, "metrics.txt"), N_particles, T)
    print(f"\nMetrics saved to {os.path.join(result_path, 'metrics.txt')}")

    # Plots

    # Combined data plot
    plot_data_combined(t, [
        (xs_log, ys_log, 'Log-Transformed (Linear): $z_t = \\log(y_t^2)$'),
        (xs_add, ys_add, 'Additive Noise (Nonlinear): $y_t = \\beta \\exp(x_t/2) + w_t$'),
        (xs_strong, ys_strong, 'Additive Noise (Strong Nonlinear): $y_t = \\beta \\exp(2x_t) + w_t$', True),
    ], os.path.join(result_path, "1_data_all_models.pdf"))

    plot_filtering_results(t, xs_log, results_log,
                           'Log-Transformed (Linear)',
                           os.path.join(result_path, "2_log_transformed_results.pdf"))

    plot_filtering_results(t, xs_add, results_add,
                           'Additive Noise (Nonlinear)',
                           os.path.join(result_path, "3_additive_noise_results.pdf"))

    plot_filtering_results(t, xs_strong, results_strong,
                           'Additive Noise (Strong Nonlinear)',
                           os.path.join(result_path, "4_strong_nonlinear_results.pdf"))


if __name__ == "__main__":
    seed = 42
    rng_np = np.random.default_rng(seed)

    result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part1', 'exp_1_2_stochastic_volatility')
    os.makedirs(result_path, exist_ok=True)

    start_time = time.time()
    run_experiment(rng_np, result_path, T=500, N_particles=int(1e5))
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
