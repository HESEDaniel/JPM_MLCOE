"""Li et al. (2017) multi-target acoustic tracking experiment."""
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.filters.ekf import extended_kalman_filter
from src.filters.ukf import unscented_kalman_filter
from src.filters.pf import particle_filter
from src.flows.edh import exact_daum_huang_flow
from src.flows.ledh import local_edh_flow
from src.flows.pfpf_edh import pfpf_edh
from src.flows.pfpf_ledh import pfpf_ledh
from src.utils import exponential_lambda_schedule
from src.utils.experiment_logger import ExperimentLogger
from src.utils.visualization import (
    plot_omat_comparison,
    plot_ess_comparison,
    plot_omat_boxplot,
)

# Import SSM module
from src.ssm.multi_target_acoustic import (
    MultiTargetAcousticModel,
    multi_target_acoustic_ssm,
    sample_initial_distribution,
    compute_omat_trajectory,
    DEFAULT_N_TARGETS,
    DEFAULT_AREA_SIZE,
)

# --- Filter Result Container ---

@dataclass
class FilterResult:
    """Container for filter results."""
    name: str
    m_filt: np.ndarray
    P_filt: np.ndarray
    runtime: float
    ess: Optional[np.ndarray] = None
    resample_count: int = 0
    omat: Optional[np.ndarray] = None

# --- Filter Implementations ---

def run_ekf(model: MultiTargetAcousticModel, ys: np.ndarray,
            m0: np.ndarray, P0: np.ndarray) -> FilterResult:
    """Run Extended Kalman Filter."""
    start = time.time()
    m_filt, P_filt, _ = extended_kalman_filter(
        f=model.f,
        h=model.h,
        F_jacobian=model.f_jacobian,
        H_jacobian=model.h_jacobian,
        Q=model.Q_filt, R=model.R, m0=m0, P0=P0, ys=ys, joseph=True
    )
    return FilterResult('EKF', m_filt, P_filt, time.time() - start)

def run_ukf(model: MultiTargetAcousticModel, ys: np.ndarray,
            m0: np.ndarray, P0: np.ndarray) -> FilterResult:
    """
    Run Unscented Kalman Filter.

    For 16D state space, use appropriate UKF parameters:
    - alpha_ukf: larger value (0.1) for better numerical stability
    - kappa: 3 - n_x = -13 (or 0) for high-dimensional problems
    """
    start = time.time()
    m_filt, P_filt, _ = unscented_kalman_filter(
        f=model.f,
        h=model.h,
        Q=model.Q_filt, R=model.R, m0=m0, P0=P0, ys=ys,
        alpha=0.1,
        kappa=0.0,
        beta=2.0,
        joseph=True
    )
    return FilterResult('UKF', m_filt, P_filt, time.time() - start)

def run_bpf(model: MultiTargetAcousticModel, ys: np.ndarray,
            m0: np.ndarray, P0: np.ndarray, N_particles: int,
            rng: np.random.Generator, name: Optional[str] = None) -> FilterResult:
    """Run Bootstrap Particle Filter."""
    start = time.time()
    m_filt, P_filt, ess, resample_count = particle_filter(
        model.f, model.h, model.Q_sampler, model.log_likelihood,
        m0, P0, ys, N_particles=N_particles, resample_threshold=0.5, rng=rng
    )

    if name is None:
        if N_particles == 100000:
            name = 'BPF(100K)'
        elif N_particles == 1000000:
            name = 'BPF(1M)'
        else:
            name = 'BPF'

    return FilterResult(name, m_filt, P_filt, time.time() - start, ess, resample_count)

def run_edh(model: MultiTargetAcousticModel, ys: np.ndarray,
            m0: np.ndarray, P0: np.ndarray, N_particles: int,
            n_flow_steps: int, rng: np.random.Generator) -> FilterResult:
    """Run standalone EDH flow filter (no importance weights)."""
    T = len(ys)
    lambda_schedule = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)

    particles = rng.multivariate_normal(m0, P0, size=N_particles)
    m_filt = np.zeros((T, model.n_x))
    P_filt = np.zeros((T, model.n_x, model.n_x))
    m, P = m0, P0

    start = time.time()
    for t in range(T):
        particles, _, _, m, P = exact_daum_huang_flow(
            particles, m, P, model.f, model.f_jacobian, model.Q_filt,
            model.h, model.h_jacobian, model.R, ys[t],
            lambda_schedule=lambda_schedule, redraw=True, rng=rng, filter_type='ekf'
        )

        m_filt[t] = np.mean(particles, axis=0)
        diff = particles - m_filt[t]
        P_filt[t] = (diff.T @ diff) / (N_particles - 1)

    return FilterResult('EDH', m_filt, P_filt, time.time() - start)

def run_ledh(model: MultiTargetAcousticModel, ys: np.ndarray,
             m0: np.ndarray, P0: np.ndarray, N_particles: int,
             n_flow_steps: int, rng: np.random.Generator) -> FilterResult:
    """Run standalone LEDH flow filter (no importance weights)."""
    T = len(ys)
    lambda_schedule = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)

    particles = rng.multivariate_normal(m0, P0, size=N_particles)
    m_filt = np.zeros((T, model.n_x))
    P_filt = np.zeros((T, model.n_x, model.n_x))
    m, P = m0, P0

    start = time.time()
    for t in range(T):
        particles, _, _, m, P = local_edh_flow(
            particles, m, P, model.f, model.f_jacobian, model.Q_filt,
            model.h, model.h_jacobian, model.R, ys[t],
            lambda_schedule=lambda_schedule, store_history=False,
            redraw=True, rng=rng, filter_type='ekf'
        )

        m_filt[t] = np.mean(particles, axis=0)
        diff = particles - m_filt[t]
        P_filt[t] = (diff.T @ diff) / (N_particles - 1)

    return FilterResult('LEDH', m_filt, P_filt, time.time() - start)

def run_pf_pf_edh(model: MultiTargetAcousticModel, ys: np.ndarray,
                  m0: np.ndarray, P0: np.ndarray, N_particles: int,
                  n_flow_steps: int, rng: np.random.Generator) -> FilterResult:
    """Run PF-PF with EDH flow (Algorithm 2 from Li et al. 2017)."""
    lambda_schedule = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)

    start = time.time()
    m_filt, P_filt, ess, resample_count, _ = pfpf_edh(
        f=model.f,
        h=model.h,
        H_jac=model.h_jacobian,
        Q=model.Q_filt,
        R=model.R,
        m0=m0,
        P0=P0,
        ys=ys,
        N_particles=N_particles,
        lambda_schedule=lambda_schedule,
        resample_threshold=0.5,
        filter_type='ekf',
        F_jacobian=model.f_jacobian,
        rng=rng
    )
    return FilterResult('PF-PF(EDH)', m_filt, P_filt, time.time() - start, ess, resample_count)

def run_pf_pf_ledh(model: MultiTargetAcousticModel, ys: np.ndarray,
                   m0: np.ndarray, P0: np.ndarray, N_particles: int,
                   n_flow_steps: int, rng: np.random.Generator) -> FilterResult:
    """Run PF-PF with LEDH flow (Algorithm 1 from Li et al. 2017)."""
    lambda_schedule = exponential_lambda_schedule(n_steps=n_flow_steps, ratio=1.2)

    start = time.time()
    m_filt, P_filt, ess, resample_count, _ = pfpf_ledh(
        f=model.f,
        h=model.h,
        H_jac=model.h_jacobian,
        Q=model.Q_filt,
        R=model.R,
        m0=m0,
        P0=P0,
        ys=ys,
        N_particles=N_particles,
        lambda_schedule=lambda_schedule,
        resample_threshold=0.5,
        filter_type='ekf',
        F_jacobian=model.f_jacobian,
        rng=rng
    )
    return FilterResult('PF-PF(LEDH)', m_filt, P_filt, time.time() - start, ess, resample_count)

# --- Visualization ---

def plot_single_trajectory(xs_true: np.ndarray, result: FilterResult,
                           model: MultiTargetAcousticModel, save_path: str) -> None:
    """Plot trajectory for a single algorithm."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Sensors
    ax.scatter(model.sensor_coords[:, 0], model.sensor_coords[:, 1],
               c='blue', s=30, marker='s', label='Sensors', zorder=5)

    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']

    # Extract all positions: [T, N_TARGETS, 2]
    true_all = xs_true.reshape(len(xs_true), model.n_targets, 4)[:, :, :2]
    est_all = result.m_filt.reshape(len(result.m_filt), model.n_targets, 4)[:, :, :2]

    # Match estimates to true targets at each time step
    est_all_matched = np.zeros_like(est_all)
    for t in range(len(est_all)):
        diff = true_all[t][:, np.newaxis, :] - est_all[t][np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        _, assignment = linear_sum_assignment(distances)
        est_all_matched[t] = est_all[t, assignment, :]

    for c in range(model.n_targets):
        true_positions = true_all[:, c, :]
        est_positions = est_all_matched[:, c, :]

        ax.plot(true_positions[:, 0], true_positions[:, 1], '-', color=colors[c],
                linewidth=2, label=f'Target {c+1} (true)')
        ax.plot(est_positions[:, 0], est_positions[:, 1], '--', color=colors[c],
                linewidth=1.5, alpha=0.8, label=f'Target {c+1} (est)')
        ax.scatter(true_positions[0, 0], true_positions[0, 1], marker='x',
                   color=colors[c], s=80, zorder=10)

    ax.set_xlim(0, model.area_size)
    ax.set_ylim(0, model.area_size)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{result.name} - Trajectory Estimation', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def save_table1(metrics: Dict[str, Dict], save_path: str) -> None:
    """Save Table 1 results to file."""
    with open(save_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('Table 1: Performance Comparison (Li et al. 2017, Section 5A)\n')
        f.write('=' * 80 + '\n\n')

        header = f"{'Algorithm':<15} {'Avg OMAT (m)':>12} {'Std OMAT':>12} {'Avg ESS':>12} {'Resamples':>12} {'Runtime':>12}"
        f.write(header + '\n')
        f.write('-' * 80 + '\n')

        for name, m in metrics.items():
            ess_str = f"{m.get('mean_ess', 0):.1f}" if 'mean_ess' in m else 'N/A'
            resamp_str = f"{m.get('mean_resamples', 0):.1f}" if 'mean_resamples' in m else 'N/A'
            line = f"{name:<15} {m['mean_omat']:>12.3f} {m['std_omat']:>12.3f} {ess_str:>12} {resamp_str:>12} {m['mean_runtime']:>11.2f}s"
            f.write(line + '\n')

        f.write('=' * 80 + '\n')

    print(f'Table 1 saved to: {save_path}')

# --- Main Experiment ---

def run_single_trial(model: MultiTargetAcousticModel, rng: np.random.Generator,
                     xs_true: np.ndarray, ys: np.ndarray,
                     N_particles: int, n_flow_steps: int,
                     algorithms: List[str]) -> Dict:
    """Run a single trial with specified algorithms."""
    m0, P0 = sample_initial_distribution(rng, model.n_targets, model.area_size)

    results = {}

    for algo in algorithms:
        if algo == 'EKF':
            results[algo] = run_ekf(model, ys, m0, P0)
        elif algo == 'UKF':
            results[algo] = run_ukf(model, ys, m0, P0)
        elif algo == 'BPF':
            results[algo] = run_bpf(model, ys, m0, P0, N_particles, rng, name='BPF')
        elif algo == 'BPF(100K)':
            results[algo] = run_bpf(model, ys, m0, P0, 100000, rng, name='BPF(100K)')
        elif algo == 'BPF(1M)':
            results[algo] = run_bpf(model, ys, m0, P0, 1000000, rng, name='BPF(1M)')
        elif algo == 'EDH':
            results[algo] = run_edh(model, ys, m0, P0, N_particles, n_flow_steps, rng)
        elif algo == 'LEDH':
            results[algo] = run_ledh(model, ys, m0, P0, N_particles, n_flow_steps, rng)
        elif algo == 'PF-PF(EDH)':
            results[algo] = run_pf_pf_edh(model, ys, m0, P0, N_particles, n_flow_steps, rng)
        elif algo == 'PF-PF(LEDH)':
            results[algo] = run_pf_pf_ledh(model, ys, m0, P0, N_particles, n_flow_steps, rng)

    # Compute OMAT for each result
    for result in results.values():
        result.omat = compute_omat_trajectory(xs_true, result.m_filt, model.n_targets)

    return {'xs_true': xs_true, 'ys': ys, 'm0': m0, 'P0': P0, 'results': results}

def run_experiment(
    T: int = 100,
    N_particles: int = 500,
    n_flow_steps: int = 20,
    n_trajectories: int = 100,
    n_trials_per_trajectory: int = 5,
    seed: int = 42,
    result_dir: Optional[str] = None,
    algorithms: Optional[List[str]] = None,
    force_rerun: bool = False
) -> Dict:
    """
    Run the full Li(17) Section 5A replication experiment.

    Parameters
    ----------
    T : int
        Number of time steps (measurements per trajectory)
    N_particles : int
        Number of particles (paper uses 500)
    n_flow_steps : int
        Number of flow integration steps
    n_trajectories : int
        Number of random trajectories (paper uses 100)
    n_trials_per_trajectory : int
        Number of trials per trajectory with different initial distributions (paper uses 5)
    seed : int
        Random seed
    result_dir : str
        Output directory
    algorithms : list[str], optional
        Algorithms to run. Default: all
    force_rerun : bool
        If True, ignore cached results and rerun all algorithms

    Returns
    -------
    dict
        Experiment results and metrics
    """
    if algorithms is None:
        algorithms = ['EKF', 'UKF', 'BPF', 'BPF(100K)', 'BPF(1M)', 'EDH', 'LEDH', 'PF-PF(EDH)', 'PF-PF(LEDH)']

    # Create model
    model = MultiTargetAcousticModel()

    # Initialize experiment logger
    logger = ExperimentLogger(experiment_name='exp_2_1a_li17')

    # Config for cache matching
    config = {
        'T': T,
        'N_particles': N_particles,
        'n_flow_steps': n_flow_steps,
        'n_trajectories': n_trajectories,
        'n_trials': n_trials_per_trajectory,
        'seed': seed,
    }

    # Check which algorithms are already cached
    cached_algos = [] if force_rerun else logger.get_cached_algorithms(**config)
    algos_to_run = [a for a in algorithms if a not in cached_algos]

    # Create timestamped run directory
    if result_dir is None:
        result_dir = logger.create_timestamped_run_dir()
    
    # Create output directories
    figs_dir = os.path.join(result_dir, 'figures')
    traj_dir = os.path.join(figs_dir, 'trajectories')
    metrics_dir = os.path.join(result_dir, 'metrics')
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Initialize aggregate result containers
    particle_filter_algos = ['BPF', 'BPF(100K)', 'BPF(1M)', 'PF-PF(EDH)', 'PF-PF(LEDH)']
    all_omat = {algo: [] for algo in algorithms}
    all_ess = {algo: [] for algo in algorithms if algo in particle_filter_algos}
    all_resamples = {algo: [] for algo in algorithms if algo in particle_filter_algos}
    all_runtime = {algo: [] for algo in algorithms}
    best_trial_data = {algo: None for algo in algorithms}

    # Load cached results
    for algo in cached_algos:
        if algo not in algorithms:
            continue
        cached_data = logger.load_algorithm_result(algo, **config)
        if cached_data is not None:
            all_omat[algo] = list(cached_data['all_omat'])
            all_runtime[algo] = list(cached_data['all_runtime'])
            if algo in all_ess and 'all_ess' in cached_data:
                all_ess[algo] = list(cached_data['all_ess'])
                all_resamples[algo] = list(cached_data['all_resamples'])
            if 'best_xs_true' in cached_data and 'best_m_filt' in cached_data:
                best_trial_data[algo] = {
                    'xs_true': cached_data['best_xs_true'],
                    'm_filt': cached_data['best_m_filt'],
                }

    # Run algorithms that are not cached
    if algos_to_run:
        rng = np.random.default_rng(seed)
        algo_trials = {algo: [] for algo in algos_to_run}

        total_runs = 0
        for traj_idx in range(n_trajectories):
            xs_true, ys = model.simulate(T, rng)

            for trial_idx in range(n_trials_per_trajectory):
                total_runs += 1

                trial_data = run_single_trial(model, rng, xs_true, ys,
                                              N_particles, n_flow_steps, algos_to_run)

                for algo, result in trial_data['results'].items():
                    all_omat[algo].append(result.omat)
                    all_runtime[algo].append(result.runtime)
                    if result.ess is not None:
                        all_ess[algo].append(result.ess)
                        all_resamples[algo].append(result.resample_count)

                    algo_trials[algo].append({
                        'xs_true': xs_true,
                        'm_filt': result.m_filt,
                        'omat': result.omat,
                    })

        # Save newly computed algorithm results to cache
        for algo in algos_to_run:
            trial_omat_means = [np.mean(t['omat']) for t in algo_trials[algo]]
            best_idx = np.argmin(trial_omat_means)
            best_trial = algo_trials[algo][best_idx]
            best_trial_data[algo] = {
                'xs_true': best_trial['xs_true'],
                'm_filt': best_trial['m_filt'],
            }

            cache_data = {
                'all_omat': np.array(all_omat[algo]),
                'all_runtime': np.array(all_runtime[algo]),
                'best_xs_true': best_trial['xs_true'],
                'best_m_filt': best_trial['m_filt'],
            }
            if algo in all_ess and all_ess[algo]:
                cache_data['all_ess'] = np.array(all_ess[algo])
                cache_data['all_resamples'] = np.array(all_resamples[algo])

            mean_omat = np.mean([np.mean(o) for o in all_omat[algo]])
            std_omat = np.std([np.mean(o) for o in all_omat[algo]])
            mean_runtime = np.mean(all_runtime[algo])

            algo_metrics = {
                'mean_omat': mean_omat,
                'std_omat': std_omat,
            }
            if algo in all_ess and all_ess[algo]:
                algo_metrics['mean_ess'] = np.mean([np.mean(e) for e in all_ess[algo]])
                algo_metrics['mean_resamples'] = np.mean(all_resamples[algo])

            logger.save_algorithm_result(
                algorithm=algo,
                config=config,
                data=cache_data,
                metrics=algo_metrics,
                runtime_sec=mean_runtime * len(all_runtime[algo]),
            )

    # Compute metrics for all algorithms
    metrics = {}
    for algo in algorithms:
        if not all_omat[algo]:
            continue

        mean_omat = np.mean([np.mean(o) for o in all_omat[algo]])
        std_omat = np.std([np.mean(o) for o in all_omat[algo]])
        mean_runtime = np.mean(all_runtime[algo])

        metrics[algo] = {
            'mean_omat': mean_omat,
            'std_omat': std_omat,
            'mean_runtime': mean_runtime
        }

        if algo in all_ess and all_ess[algo]:
            mean_ess = np.mean([np.mean(e) for e in all_ess[algo]])
            mean_resamples = np.mean(all_resamples[algo])
            metrics[algo]['mean_ess'] = mean_ess
            metrics[algo]['mean_resamples'] = mean_resamples
            ess_str = f'{mean_ess:.1f}'
            resamp_str = f'{mean_resamples:.1f}'
        else:
            ess_str = 'N/A'
            resamp_str = 'N/A'

        source = 'cached' if algo in cached_algos else 'computed'

    # Generate plots
    
    # Individual trajectory plots
    for algo in algorithms:
        if best_trial_data[algo] is None:
            continue
        safe_name = algo.replace('(', '_').replace(')', '')
        result = FilterResult(
            name=algo,
            m_filt=best_trial_data[algo]['m_filt'],
            P_filt=np.zeros((len(best_trial_data[algo]['m_filt']), model.n_x, model.n_x)),
            runtime=0.0,
        )
        plot_single_trajectory(
            best_trial_data[algo]['xs_true'], result, model,
            os.path.join(traj_dir, f'trajectory_{safe_name}.png')
        )
    
    # Average OMAT plot
    mean_omat_dict = {algo: np.mean(np.array(all_omat[algo]), axis=0)
                      for algo in algorithms if all_omat[algo]}
    if mean_omat_dict:
        plot_omat_comparison(mean_omat_dict, os.path.join(figs_dir, 'omat_comparison.png'))
        
    # OMAT box plot
    omat_for_boxplot = {algo: all_omat[algo] for algo in algorithms if all_omat[algo]}
    if omat_for_boxplot:
        plot_omat_boxplot(omat_for_boxplot, os.path.join(figs_dir, 'omat_boxplot.png'))
        
    # ESS plot
    if all_ess:
        mean_ess_dict = {algo: np.mean(np.array(ess), axis=0)
                         for algo, ess in all_ess.items() if ess}
        if mean_ess_dict:
            plot_ess_comparison(mean_ess_dict, N_particles,
                                os.path.join(figs_dir, 'ess_comparison.png'))
            
    # Save Table 1
    save_table1(metrics, os.path.join(metrics_dir, 'table1.txt'))

    # Save config
    with open(os.path.join(metrics_dir, 'experiment_config.txt'), 'w') as f:
        f.write('Experiment Configuration (Li et al. 2017, Section 5A)\n')
        f.write('=' * 50 + '\n')
        f.write(f'T (time steps per trajectory) = {T}\n')
        f.write(f'N_particles = {N_particles}\n')
        f.write(f'n_flow_steps = {n_flow_steps}\n')
        f.write(f'n_trajectories = {n_trajectories}\n')
        f.write(f'n_trials_per_trajectory = {n_trials_per_trajectory}\n')
        f.write(f'Total runs per algorithm = {n_trajectories * n_trials_per_trajectory}\n')
        f.write(f'seed = {seed}\n')
        f.write(f'algorithms = {algorithms}\n')
        f.write(f'\nCached algorithms: {cached_algos}\n')
        f.write(f'Computed algorithms: {algos_to_run}\n')

    return {
        'metrics': metrics,
        'all_omat': all_omat,
        'all_ess': all_ess,
        'n_trajectories': n_trajectories,
        'n_trials_per_trajectory': n_trials_per_trajectory,
    }

if __name__ == '__main__':
    results = run_experiment(
        T=40,
        N_particles=500,
        n_trajectories=5,
        n_trials_per_trajectory=100,
        n_flow_steps=29,
        seed=3239,
        algorithms=['EKF', 'UKF', 'EDH', 'LEDH', 'PF-PF(EDH)', 'PF-PF(LEDH)', 'BPF(100K)'],
        # algorithms=['PF-PF(EDH)']
    )
