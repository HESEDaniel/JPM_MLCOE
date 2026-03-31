"""Hu et al. (2021) RKHS kernel comparison experiment (TensorFlow)."""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.flows.rkhs_pff import localization_matrix, _rkhs_flow_step
from src.filters.enkf import enkf_posterior_analytical
from src.filters.common import DTYPE
from src.ssm.lorenz96 import lorenz96_step
from src.utils.visualization import plot_contours


def rkhs_particle_flow_tf(particles, h_fn, H_jac_fn, R, y,
                           n_steps=10, step_size=0.1, loc_radius=4.0,
                           bandwidth_alpha=None, kernel_type='matrix-valued',
                           adaptive_step=False, step_factor=1.4,
                           decrease_patience=20, min_step=1e-6, max_step=1.0):
    """Single-observation RKHS particle flow (TensorFlow).

    Mirrors the numpy ``rkhs_particle_flow`` but uses TF tensors and
    delegates each flow step to ``_rkhs_flow_step``.

    Parameters
    ----------
    particles : tf.Tensor [N, n_x]
    h_fn, H_jac_fn : callable
    R : tf.Tensor [n_y, n_y]
    y : tf.Tensor [n_y]
    n_steps, step_size, loc_radius, bandwidth_alpha, kernel_type,
    adaptive_step, step_factor, decrease_patience, min_step, max_step
        Same semantics as the numpy version.

    Returns
    -------
    particles : tf.Tensor [N, n_x]
    """
    N = tf.shape(particles)[0]
    n_x = tf.shape(particles)[1]
    N_f = tf.cast(N, DTYPE)

    # Ensemble covariance with localization
    x_bar = tf.reduce_mean(particles, axis=0)
    X = tf.transpose(particles - x_bar)
    B = X @ tf.transpose(X) / (N_f - 1.0)
    C_loc = localization_matrix(n_x, loc_radius)
    B_loc = B * C_loc
    B_loc_inv = tf.linalg.inv(B_loc)
    R_inv = tf.linalg.inv(R)

    alpha = bandwidth_alpha if bandwidth_alpha is not None else 1.0 / float(N_f.numpy())

    if kernel_type == 'scalar':
        bandwidths = None
    else:
        bandwidths = alpha * tf.linalg.diag_part(B_loc)

    current_step = step_size
    prev_flow_mag = float('inf')
    consecutive_decreases = 0

    for _ in range(n_steps):
        particles, flow_mag = _rkhs_flow_step(
            particles, h_fn, H_jac_fn, R, y,
            B_loc, B_loc_inv, R_inv, bandwidths,
            kernel_type, current_step)

        if adaptive_step:
            fm = float(flow_mag)
            if fm < prev_flow_mag:
                consecutive_decreases += 1
                if consecutive_decreases >= decrease_patience:
                    current_step = min(current_step * step_factor, max_step)
                    consecutive_decreases = 0
            else:
                current_step = max(current_step / step_factor, min_step)
                consecutive_decreases = 0
            prev_flow_mag = fm

    return particles


def run_experiment(seed=42):
    """Run the experiment."""
    rng = np.random.default_rng(seed)
    t_start = time.time()

    # Settings
    n_x, F, dt = 200, 8.0, 0.01
    N_p, n_flow_steps, step_size, loc_radius, eps = 20, 100, 0.05, 4.0, 0.5

    # Initial condition and spinup (numpy)
    x = np.array([F + 1 if (a + 1) % 5 == 0 else F for a in range(n_x)])
    for _ in range(1000):
        x = lorenz96_step(x, F, dt)
    x_true = x.copy()

    # Prior ensemble (numpy, then convert)
    particles_prior_np = x_true + rng.normal(0, np.sqrt(2.0), (N_p, n_x))
    particles_prior = tf.constant(particles_prior_np, dtype=DTYPE)

    # Observation model: every 4th variable
    obs_idx = np.arange(3, n_x, 4)
    n_y = len(obs_idx)
    H_np = np.zeros((n_y, n_x))
    H_np[np.arange(n_y), obs_idx] = 1.0
    H_tf = tf.constant(H_np, dtype=DTYPE)

    h_fn = lambda x: tf.linalg.matvec(H_tf, x)
    H_jac_fn = lambda x: H_tf

    R_np = eps * np.eye(n_y)
    R_tf = tf.constant(R_np, dtype=DTYPE)

    y_np = H_np @ x_true + rng.normal(0, np.sqrt(eps), n_y)
    y_tf = tf.constant(y_np, dtype=DTYPE)

    # EnKF posterior for contours
    B_np = 2.0 * np.eye(n_x)
    B_tf = tf.constant(B_np, dtype=DTYPE)
    C_tf = localization_matrix(n_x, loc_radius)

    x_bar_prior = tf.reduce_mean(particles_prior, axis=0)
    m_enkf, P_enkf = enkf_posterior_analytical(x_bar_prior, B_tf, H_tf, R_tf, y_tf, C_tf)

    t_setup = time.time()
    print(f"Setup time: {t_setup - t_start:.3f}s")

    # Run RKHS flow (matrix-valued kernel)
    flow_args = dict(n_steps=n_flow_steps, step_size=step_size,
                     loc_radius=loc_radius, adaptive_step=True)

    t0 = time.time()
    post_matrix = rkhs_particle_flow_tf(
        tf.identity(particles_prior), h_fn, H_jac_fn, R_tf, y_tf,
        kernel_type='matrix-valued', **flow_args)
    t_matrix = time.time() - t0
    print(f"Matrix-valued kernel flow: {t_matrix:.3f}s")

    # Run RKHS flow (scalar kernel)
    t0 = time.time()
    post_scalar = rkhs_particle_flow_tf(
        tf.identity(particles_prior), h_fn, H_jac_fn, R_tf, y_tf,
        kernel_type='scalar', **flow_args)
    t_scalar = time.time() - t0
    print(f"Scalar kernel flow: {t_scalar:.3f}s")

    t_total = time.time() - t_start
    print(f"Total experiment time: {t_total:.3f}s")

    # Convert to numpy for plotting
    post_matrix_np = post_matrix.numpy()
    post_scalar_np = post_scalar.numpy()
    m_enkf_np = m_enkf.numpy()
    P_enkf_np = P_enkf.numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    idx = [18, 19]

    for ax, post, title in [(axes[0], post_matrix_np, '(a) Matrix-valued kernel'),
                             (axes[1], post_scalar_np, '(b) Scalar kernel')]:
        plot_contours(ax, m_enkf_np[idx], P_enkf_np[np.ix_(idx, idx)],
                      levels=[1, 2, 3], color='gray', alpha=0.7)
        ax.scatter(particles_prior_np[:, 18], particles_prior_np[:, 19],
                   facecolors='none', edgecolors='black', s=50, label='Prior')
        ax.scatter(post[:, 18], post[:, 19],
                   c='red', s=50, label='Posterior')
        ax.set(xlabel=r'$x_{19}$ (unobserved)',
               ylabel=r'$x_{20}$ (observed)',
               title=title, aspect='equal')
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()

    result_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part1', 'exp_2_2_hu21')
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, 'fig3_kernel_comparison.pdf')
    fig.savefig(save_path)
    print(f'Saved to {save_path}')
    plt.close(fig)


if __name__ == '__main__':
    run_experiment()
