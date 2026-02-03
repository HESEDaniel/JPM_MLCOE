"""Stochastic Volatility model filter comparison."""
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm.stochastic_volatility import SVLogTransformed, SVAdditiveNoise
from src.filters.ekf import extended_kalman_filter
from src.filters.ukf import unscented_kalman_filter
from src.filters.pf import particle_filter
from src.utils.metrics import compute_mse, compute_nees

@dataclass
class ProfileResult:
    """Container for profiling results."""
    runtime_s: float = 0.0
    peak_memory_mb: float = 0.0

@contextmanager
def profile():
    """Context manager for measuring runtime and peak memory usage."""
    result = ProfileResult()
    tracemalloc.start()
    start = time.perf_counter()
    yield result
    result.runtime_s = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.peak_memory_mb = peak / (1024 * 1024)

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
    peak_memory_mb: float
    m_filt: np.ndarray = field(repr=False)
    P_filt: np.ndarray = field(repr=False)

def run_filters(model, xs, ys, N_particles=1000, rng=None) -> Dict[str, FilterMetrics]:
    """Run EKF, UKF, PF on given model and observations with full profiling."""
    if rng is None:
        rng = np.random.default_rng()

    zs = model.transform_obs(ys)
    xs_col = xs.reshape(-1, 1) if xs.ndim == 1 else xs
    results = {}

    # EKF
    with profile() as prof_ekf:
        m_ekf, P_ekf, _ = extended_kalman_filter(
            model.f, model.h, model.F_jac, model.H_jac,
            model.Q, model.R, model.m0, model.P0, zs
        )
    nees_ekf = compute_nees(m_ekf, P_ekf, xs_col)
    results['EKF'] = FilterMetrics(
        name='EKF',
        mse=compute_mse(m_ekf[:, 0], xs),
        nees_mean=np.mean(nees_ekf),
        nees_std=np.std(nees_ekf),
        ess_mean=None,
        ess_min=None,
        runtime_s=prof_ekf.runtime_s,
        peak_memory_mb=prof_ekf.peak_memory_mb,
        m_filt=m_ekf,
        P_filt=P_ekf,
    )

    # UKF
    with profile() as prof_ukf:
        m_ukf, P_ukf, _ = unscented_kalman_filter(
            model.f, model.h, model.Q, model.R, model.m0, model.P0, zs
        )
    nees_ukf = compute_nees(m_ukf, P_ukf, xs_col)
    results['UKF'] = FilterMetrics(
        name='UKF',
        mse=compute_mse(m_ukf[:, 0], xs),
        nees_mean=np.mean(nees_ukf),
        nees_std=np.std(nees_ukf),
        ess_mean=None,
        ess_min=None,
        runtime_s=prof_ukf.runtime_s,
        peak_memory_mb=prof_ukf.peak_memory_mb,
        m_filt=m_ukf,
        P_filt=P_ukf,
    )

    # PF
    with profile() as prof_pf:
        m_pf, P_pf, ess_pf, _ = particle_filter(
            f=lambda x: model.alpha * x, h=lambda x: x,
            Q_sampler=model.Q_sampler, log_likelihood=model.log_likelihood,
            m0=model.m0, P0=model.P0, ys=ys.reshape(-1, 1),
            N_particles=N_particles, resample_threshold=0.5, rng=rng
        )
    nees_pf = compute_nees(m_pf, P_pf, xs_col)
    results['PF'] = FilterMetrics(
        name='PF',
        mse=compute_mse(m_pf[:, 0], xs),
        nees_mean=np.mean(nees_pf),
        nees_std=np.std(nees_pf),
        ess_mean=np.mean(ess_pf),
        ess_min=np.min(ess_pf),
        runtime_s=prof_pf.runtime_s,
        peak_memory_mb=prof_pf.peak_memory_mb,
        m_filt=m_pf,
        P_filt=P_pf,
    )

    return results

def write_metrics_table(all_results: Dict[str, Dict[str, FilterMetrics]],
                        save_path: str, N_particles: int, T: int):
    """Write evaluation metrics to a formatted text file."""
    with open(save_path, 'w') as f:
        f.write("Stochastic Volatility Filter Comparison\n")
        f.write(f"Configuration: T={T}, N_particles={N_particles}\n\n")

        header = f"{'Model':<25} {'Filter':<6} {'MSE':>10} {'NEES':>12} {'ESS_mean':>10} {'ESS_min':>10} {'Runtime':>10} {'Memory':>10}\n"
        units  = f"{'':25} {'':6} {'':>10} {'(mean+/-std)':>12} {'':>10} {'':>10} {'(s)':>10} {'(MB)':>10}\n"

        f.write(header)
        f.write(units)

        for model_name, results in all_results.items():
            for filter_name, m in results.items():
                ess_mean_str = f"{m.ess_mean:.1f}" if m.ess_mean is not None else "N/A"
                ess_min_str = f"{m.ess_min:.1f}" if m.ess_min is not None else "N/A"
                nees_str = f"{m.nees_mean:.2f}+/-{m.nees_std:.2f}"

                f.write(f"{model_name:<25} {filter_name:<6} {m.mse:>10.4f} {nees_str:>12} "
                        f"{ess_mean_str:>10} {ess_min_str:>10} {m.runtime_s:>10.4f} {m.peak_memory_mb:>10.2f}\n")
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

def run_sis_capture_weights(model, ys, N_particles, capture_times, rng):
    """
    Run Sequential Importance Sampling (SIS) without resampling to capture
    weight degeneracy at specific time steps.

    Returns weights at capture_times for histogram visualization.
    """
    T = len(ys)
    N = N_particles

    # Initialize particles
    particles = rng.multivariate_normal(model.m0, model.P0, size=N)
    log_w = np.zeros(N)

    weights_at_times = {}

    for t in range(T):
        # Predict
        noise = model.Q_sampler(rng, N)
        particles = model.alpha * particles + noise

        # Update weights (accumulate log weights - no resampling)
        # Wrap scalar observation in array for log_likelihood interface
        y_t = np.array([ys[t]]) if np.isscalar(ys[t]) else ys[t]
        log_lik = model.log_likelihood(y_t, particles)
        log_w += log_lik

        # Normalize for storage
        log_w_normalized = log_w - np.max(log_w)
        w = np.exp(log_w_normalized)
        w /= w.sum()

        # Capture weights at specified times
        if t + 1 in capture_times:  # t+1 because we want t=1,2,...
            weights_at_times[t + 1] = w.copy()

    return weights_at_times

def run_pf_capture_weights(model, ys, N_particles, capture_times, rng,
                           resample_threshold=0.5):
    """
    Run Particle Filter (PF) with resampling to capture weight distributions
    at specific time steps.

    Returns weights at capture_times for histogram visualization.
    """
    T = len(ys)
    N = N_particles

    # Initialize particles
    particles = rng.multivariate_normal(model.m0, model.P0, size=N)
    log_w = np.zeros(N)
    w = np.ones(N) / N

    weights_at_times = {}

    for t in range(T):
        # Predict
        noise = model.Q_sampler(rng, N)
        particles = model.alpha * particles + noise

        # Update weights
        y_t = np.array([ys[t]]) if np.isscalar(ys[t]) else ys[t]
        log_lik = model.log_likelihood(y_t, particles)
        log_w = np.log(w + 1e-300) + log_lik

        # Normalize
        log_w_normalized = log_w - np.max(log_w)
        w = np.exp(log_w_normalized)
        w /= w.sum()

        # Capture weights at specified times (before resampling)
        if t + 1 in capture_times:
            weights_at_times[t + 1] = w.copy()

        # Resample if ESS below threshold
        ess = 1.0 / np.sum(w ** 2)
        if ess < resample_threshold * N:
            # Systematic resampling
            cumsum = np.cumsum(w)
            u = (np.arange(N) + rng.uniform()) / N
            indices = np.searchsorted(cumsum, u)
            particles = particles[indices]
            w = np.ones(N) / N

    return weights_at_times

def plot_weight_histograms(weights_dict, N_particles, save_path, title=None):
    """
    Plot weight histograms at different time steps (Doucet-style Figure 3).

    Shows how particle weights are distributed over time.
    """
    from matplotlib.ticker import ScalarFormatter, LogFormatterSciNotation

    times = sorted(weights_dict.keys())
    n_times = len(times)

    fig, axes = plt.subplots(1, n_times, figsize=(4 * n_times, 4), sharey=True)
    if n_times == 1:
        axes = [axes]

    for ax, t in zip(axes, times):
        w = weights_dict[t]

        # Histogram of normalized weights (count, not density)
        ax.hist(w, bins=50, density=False, alpha=0.7,
                color='steelblue', edgecolor='black', linewidth=0.5)

        # Compute ESS for annotation
        ess = 1.0 / np.sum(w ** 2)
        max_w = np.max(w)

        ax.set_xlabel('Weight $w_t^{(i)}$')
        ax.set_title(f't = {t}\nESS = {ess:.0f}, max(w) = {max_w:.2e}')
        ax.set_xlim(0, None)

        # Use scientific notation for x-axis
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(9)

    axes[0].set_ylabel('Particle Count')
    if title is None:
        title = f'Weight Distribution (N = {N_particles})'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_experiment(rng, result_path, T=200, N_particles=1000,
                   alpha=0.91, sigma=1.0, beta=0.5, obs_std=0.5):
    """Run filter comparison experiment."""
    
    t = np.arange(T)

    # Create models
    model_log = SVLogTransformed(alpha, sigma, beta)
    model_add = SVAdditiveNoise(alpha, sigma, beta, obs_std, exp_scale=0.5)
    model_strong = SVAdditiveNoise(alpha, sigma, beta, obs_std, exp_scale=2.0)

    # Generate data
    xs_log, ys_log = model_log.simulate(T, rng)
    xs_add, ys_add = model_add.simulate(T, rng)
    xs_strong, ys_strong = model_strong.simulate(T, rng)

    # Run filters with profiling (runtime + memory + metrics)
    results_log = run_filters(model_log, xs_log, ys_log, N_particles, rng)

    results_add = run_filters(model_add, xs_add, ys_add, N_particles, rng)

    results_strong = run_filters(model_strong, xs_strong, ys_strong, N_particles, rng)

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

    # Weight histogram plots
    capture_times = [2, 5, 50]
    N_vis = 1000  # Use fewer particles for clearer visualization

    # SIS (no resampling) - shows weight degeneracy
    weights_sis = run_sis_capture_weights(
        model_strong, ys_strong, N_vis, capture_times,
        rng=np.random.default_rng(42)
    )
    plot_weight_histograms(
        weights_sis, N_vis,
        os.path.join(result_path, "5_weight_histograms_pf_noresmp.pdf"),
        title=f'PF Weight Degeneracy (N = {N_vis}, no resampling)'
    )

    # PF (with resampling) - shows effect of resampling
    weights_pf = run_pf_capture_weights(
        model_strong, ys_strong, N_vis, capture_times,
        rng=np.random.default_rng(42)
    )
    plot_weight_histograms(
        weights_pf, N_vis,
        os.path.join(result_path, "6_weight_histograms_pf_resmp.pdf"),
        title=f'PF Weight Distribution (N = {N_vis}, with resampling)'
    )


if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed)

    result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp_1_2_stochastic_volatility')
    os.makedirs(result_path, exist_ok=True)

    start_time = time.time()
    run_experiment(rng, result_path, T=500, N_particles=int(1e5))
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
