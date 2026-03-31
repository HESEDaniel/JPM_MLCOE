"""Range-Bearing model filter comparison (TensorFlow)."""
import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DTYPE = tf.float64

from src.ssm import RangeBearing
from src.filters import ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter


def run_filters(model, ys, N_particles=1000, rng_seed=42):
    """Run EKF, UKF, PF on given model and observations with timing."""
    ys_tf = tf.constant(ys, dtype=DTYPE)
    results = {}

    # EKF (with angle wrapping on bearing index 1)
    ekf = ExtendedKalmanFilter(angle_indices=[1])
    t0 = time.perf_counter()
    ekf_result = ekf.filter(model, ys_tf)
    runtime_ekf = time.perf_counter() - t0
    results['EKF'] = {
        'm': ekf_result.m_filt.numpy(),
        'P': ekf_result.P_filt.numpy(),
        'cond': ekf_result.diagnostics['cond_nums'].numpy(),
        'runtime': runtime_ekf,
    }

    # UKF
    ukf = UnscentedKalmanFilter()
    t0 = time.perf_counter()
    ukf_result = ukf.filter(model, ys_tf)
    runtime_ukf = time.perf_counter() - t0
    results['UKF'] = {
        'm': ukf_result.m_filt.numpy(),
        'P': ukf_result.P_filt.numpy(),
        'cond': ukf_result.diagnostics['cond_nums'].numpy(),
        'runtime': runtime_ukf,
    }

    # PF
    pf = ParticleFilter(n_particles=N_particles)
    rng_tf = tf.random.Generator.from_seed(rng_seed)
    t0 = time.perf_counter()
    pf_result = pf.filter(model, ys_tf, rng=rng_tf)
    runtime_pf = time.perf_counter() - t0
    results['PF'] = {
        'm': pf_result.m_filt.numpy(),
        'P': pf_result.P_filt.numpy(),
        'ess': pf_result.diagnostics['ess'].numpy(),
        'runtime': runtime_pf,
    }

    return results


def compute_position_mse(m, xs):
    """Compute position MSE: mean of squared position errors."""
    return np.mean(np.sum((xs[:, [0, 2]] - m[:, [0, 2]]) ** 2, axis=1))


def plot_data(t, xs, ys, sensor_pos, title, save_path):
    """Plot true trajectory and observations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Trajectory plot
    ax = axes[0]
    ax.plot(xs[:, 0], xs[:, 2], 'b-', lw=2, label='True Trajectory')
    ax.scatter(xs[0, 0], xs[0, 2], s=100, c='green', marker='o', zorder=5, label='Start')
    ax.scatter(xs[-1, 0], xs[-1, 2], s=100, c='red', marker='x', zorder=5, label='End')
    ax.scatter(sensor_pos[0], sensor_pos[1], s=150, c='black', marker='s', zorder=5, label='Sensor')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'{title} - Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Observations plot
    ax = axes[1]
    ax.plot(t, ys[:, 0], 'r-', lw=1, alpha=0.7, label='Range')
    ax2 = ax.twinx()
    ax2.plot(t, np.degrees(ys[:, 1]), 'b-', lw=1, alpha=0.7, label='Bearing')
    ax.set_xlabel('Time')
    ax.set_ylabel('Range', color='r')
    ax2.set_ylabel('Bearing (deg)', color='b')
    ax.set_title(f'{title} - Observations')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_filtering_results(t, xs, results, title, save_path):
    """Plot filtering results comparing true state vs estimates."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, (name, res) in zip(axes, results.items()):
        m = res['m']
        P = res['P']

        # Position error
        pos_err = np.sqrt((m[:, 0] - xs[:, 0]) ** 2 + (m[:, 2] - xs[:, 2]) ** 2)
        pos_std = np.sqrt(P[:, 0, 0] + P[:, 2, 2])

        ax.plot(t, pos_err, 'r-', label=f'{name} Position Error')
        ax.fill_between(t, np.zeros_like(pos_std), pos_std,
                        alpha=0.2, color='blue', label='Position Std')

        mse = compute_position_mse(m, xs)
        ax.set_ylabel('Position Error')
        ax.set_title(f'{name} (Pos MSE={mse:.4f}, time={res["runtime"]:.3f}s)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_trajectory_comparison(xs, results, sensor_pos, title, save_path):
    """Plot trajectory comparison for all filters."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(xs[:, 0], xs[:, 2], 'k-', lw=2, label='True', alpha=0.8)

    colors = {'EKF': 'blue', 'UKF': 'green', 'PF': 'red'}
    for name, res in results.items():
        m = res['m']
        mse = compute_position_mse(m, xs)
        ax.plot(m[:, 0], m[:, 2], '--', lw=1.5, color=colors[name],
                label=f'{name} (MSE={mse:.4f})', alpha=0.8)

    ax.scatter(sensor_pos[0], sensor_pos[1], s=20, c='black', marker='s', zorder=5, label='Sensor')
    ax.scatter(xs[0, 0], xs[0, 2], s=10, c='green', marker='o', zorder=5, label='Start')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment(rng, result_path, T=200, N_particles=1000):
    """Run filter comparison experiment."""

    t = np.arange(T)

    # Scenario 1: Far from sensor (mild nonlinearity)
    model_mild = RangeBearing(q=0.05, r_range=0.1, r_bearing=0.02)
    model_mild.set_initial(
        m0=np.array([10.0, 0.2, 10.0, 0.2]),
        P0=np.diag([0.5, 0.1, 0.5, 0.1])
    )

    # Scenario 2: Target crossing near sensor origin
    # Trajectory passes close to sensor where atan2/sqrt Jacobians are ill-conditioned
    # Large uncertainty + close proximity = EKF/UKF linearization failure
    model_extreme = RangeBearing(q=1.0, r_range=0.01, r_bearing=0.002)
    model_extreme.set_initial(
        m0=np.array([0.01, -0.01, 0.01, -0.01]),  # Start close to sensor, moving toward it
        P0=np.diag([5, 1, 5, 1])
    )

    # Generate data
    xs_mild, ys_mild = model_mild.simulate(T, rng)
    xs_extreme, ys_extreme = model_extreme.simulate(T, rng)

    # Run filters
    print("  Running filters on mild nonlinearity scenario...")
    results_mild = run_filters(model_mild, ys_mild, N_particles, rng_seed=42)

    print("  Running filters on extreme nonlinearity scenario...")
    results_extreme = run_filters(model_extreme, ys_extreme, N_particles, rng_seed=43)

    # Plots

    sensor_pos_mild = model_mild._sensor_pos.numpy()
    sensor_pos_extreme = model_extreme._sensor_pos.numpy()

    plot_data(t, xs_mild, ys_mild, sensor_pos_mild,
              'Mild Nonlinearity',
              os.path.join(result_path, "1_data_mild.png"))

    plot_data(t, xs_extreme, ys_extreme, sensor_pos_extreme,
              'Strong Nonlinearity',
              os.path.join(result_path, "2_data_extreme.png"))

    plot_filtering_results(t, xs_mild, results_mild,
                           'Mild Nonlinearity',
                           os.path.join(result_path, "3_mild_results.png"))

    plot_filtering_results(t, xs_extreme, results_extreme,
                           'Strong Nonlinearity',
                           os.path.join(result_path, "4_extreme_results.png"))

    plot_trajectory_comparison(xs_mild, results_mild, sensor_pos_mild,
                               'Mild Nonlinearity - Trajectory',
                               os.path.join(result_path, "5_mild_trajectory.png"))

    plot_trajectory_comparison(xs_extreme, results_extreme, sensor_pos_extreme,
                               'Strong Nonlinearity - Trajectory',
                               os.path.join(result_path, "6_extreme_trajectory.png"))

    # Summary
    print("\n--- Summary ---")
    print(f"{'Scenario':<25} {'Filter':<6} {'Pos MSE':>10} {'Runtime(s)':>12}")
    print("-" * 55)
    for scenario_name, xs, res_dict in [
        ('Mild Nonlinearity', xs_mild, results_mild),
        ('Strong Nonlinearity', xs_extreme, results_extreme),
    ]:
        for name, res in res_dict.items():
            mse = compute_position_mse(res['m'], xs)
            print(f"{scenario_name:<25} {name:<6} {mse:>10.4f} {res['runtime']:>12.3f}")


if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed)

    result_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part1', 'exp_1_3_range_bearing')
    os.makedirs(result_path, exist_ok=True)

    start_time = time.time()
    run_experiment(rng, result_path, T=200, N_particles=int(500))
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
