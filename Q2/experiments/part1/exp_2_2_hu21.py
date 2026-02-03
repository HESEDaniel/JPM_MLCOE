"""Hu et al. (2021) RKHS kernel comparison experiment."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.flows.rkhs_pff import rkhs_particle_flow, localization_matrix
from src.filters.enkf import enkf_posterior_analytical
from src.ssm.lorenz96 import lorenz96_step
from src.utils.visualization import plot_contours

def run_experiment(seed=42):
    """Run the experiment."""
    rng = np.random.default_rng(seed)

    # Settings
    n_x, F, dt = 200, 8.0, 0.01
    N_p, n_flow_steps, step_size, loc_radius, eps = 20, 100, 0.05, 4.0, 0.5

    # Initial condition and spinup
    x = np.array([F + 1 if (a + 1) % 5 == 0 else F for a in range(n_x)])
    for _ in range(1000):
        x = lorenz96_step(x, F, dt)
    x_true = x.copy()

    # Prior ensemble
    particles_prior = x_true + rng.normal(0, np.sqrt(2.0), (N_p, n_x))

    # Observation model: every 4th variable
    obs_idx = np.arange(3, n_x, 4)
    n_y = len(obs_idx)
    H = np.zeros((n_y, n_x))
    H[np.arange(n_y), obs_idx] = 1.0

    h, H_jac = lambda x: H @ x, lambda x: H
    R = eps * np.eye(n_y)
    y = h(x_true) + rng.normal(0, np.sqrt(eps), n_y)

    # EnKF posterior for contours
    B = 2.0 * np.eye(n_x)
    C = localization_matrix(n_x, loc_radius)
    m_enkf, P_enkf = enkf_posterior_analytical(np.mean(particles_prior, axis=0), B, H, R, y, C)

    # Run RKHS flow
    flow_args = dict(n_steps=n_flow_steps, step_size=step_size, loc_radius=loc_radius, adaptive_step=True)
    post_matrix = rkhs_particle_flow(particles_prior.copy(), h, H_jac, R, y, kernel_type='matrix-valued', **flow_args)
    post_scalar = rkhs_particle_flow(particles_prior.copy(), h, H_jac, R, y, kernel_type='scalar', **flow_args)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    idx = [18, 19]

    for ax, post, title in [(axes[0], post_matrix, '(a) Matrix-valued kernel'),
                            (axes[1], post_scalar, '(b) Scalar kernel')]:
        plot_contours(ax, m_enkf[idx], P_enkf[np.ix_(idx, idx)], levels=[1, 2, 3], color='gray', alpha=0.7)
        ax.scatter(particles_prior[:, 18], particles_prior[:, 19], facecolors='none', edgecolors='black', s=50, label='Prior')
        ax.scatter(post[:, 18], post[:, 19], c='red', s=50, label='Posterior')
        ax.set(xlabel=r'$x_{19}$ (unobserved)', ylabel=r'$x_{20}$ (observed)', title=title, aspect='equal')
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()

    result_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp_2_2_hu21')
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, 'fig3_kernel_comparison.pdf')
    fig.savefig(save_path)
    print(f'Saved to {save_path}')
    plt.close(fig)

if __name__ == '__main__':
    run_experiment()
