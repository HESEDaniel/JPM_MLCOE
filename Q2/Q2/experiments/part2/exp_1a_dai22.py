"""Dai(22) Table 1 Replication: Stochastic Particle Flow with Optimal Homotopy.

Compares straight-line beta=lam vs optimal beta* for stiffness mitigation.
Outputs:
  - results/exp_dai22/config.csv          -- experiment config
  - results/exp_dai22/table1.csv          -- per-MC MSE & tr(P)  (cf. Table 1)
  - results/exp_dai22/homotopy.csv        -- beta*(lam), beta_dot*(lam), e(lam)  (cf. Figure 2a-c)
  - results/exp_dai22/stiffness.csv       -- R_stiff(lam) for beta and beta*  (cf. Figure 2d)

Usage:
    python experiments/part2/exp_dai22.py
    python experiments/part2/exp_dai22.py --n_steps 100 --n_mc 20
"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm.bearing_only import BearingOnly2D
from src.flows.stochastic_flow import StochasticFlow, solve_optimal_homotopy
from src.filters.common import DTYPE
from src.utils.config_manager import check_and_save

RESULTS_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part2')


def compute_stiffness_ratio(ssm, P_prior_inv, neg_hess_logh, Q_diff, lam_grid,
                            beta_sched, bdot_sched):
    """Compute R_stiff(lam) = |lam_max(F)| / |lam_min(F)| along homotopy path.

    F(lam) is the Jacobian of the SDE drift (Eq 22).

    Parameters
    ----------
    ssm : BearingOnly2D
    P_prior_inv : tf.Tensor [n_x, n_x]
    neg_hess_logh : tf.Tensor [n_x, n_x]
    Q_diff : tf.Tensor [n_x, n_x]
    lam_grid : tf.Tensor [n_grid]
    beta_sched : tf.Tensor [n_grid]
    bdot_sched : tf.Tensor [n_grid]

    Returns
    -------
    r_stiff : np.ndarray [n_grid]
    """
    n_grid = len(lam_grid)
    r_stiff = np.zeros(n_grid)

    # Hessians (constant under Gaussian assumption)
    hess_logp0 = -P_prior_inv              # grad^2(log p_0) = -P^-1
    hess_logh = -neg_hess_logh             # grad^2(log h) = -H^T R^-1 H

    for j in range(n_grid):
        beta = float(beta_sched[j])
        bdot = float(bdot_sched[j])

        # grad^2(log p) = grad^2(log p_0) + beta * grad^2(log h)  (Eq 17, with sign)
        hess_logp = hess_logp0 + beta * hess_logh
        hess_logp_inv = tf.linalg.inv(hess_logp)

        # F = 0.5*Q*(grad^2 log p) - (beta_dot/2)*(grad^2 log p)^-1*(grad^2 log h)
        # Derived from K_1*(grad^2 log p) + K_2*(grad^2 log h) (Eq 13) with alpha+beta=1.
        # Note: paper's Eq (22) has an extra (grad^2 log p)^-1 -- appears to be a typo.
        F = (0.5 * Q_diff @ hess_logp
             - (bdot / 2.0) * hess_logp_inv @ hess_logh)

        # F is generally non-symmetric -> use eigvals, take |Re(eig)|
        eigs = tf.linalg.eigvals(tf.cast(F, tf.complex128)).numpy()
        re_eigs = np.abs(np.real(eigs))
        if np.min(re_eigs) > 1e-15:
            r_stiff[j] = np.max(re_eigs) / np.min(re_eigs)
        else:
            r_stiff[j] = np.inf

    return r_stiff


def run_experiment(n_mc=20, N=50, n_steps=29, mu=0.2):
    """Replicate Dai(22) Table 1 and Figure 2."""

    results_dir = os.path.join(RESULTS_BASE, 'exp_dai22', f'steps{n_steps}')

    ssm = BearingOnly2D()
    z = ssm.generate_fixed_measurement()
    Q_diff = tf.constant([[4.0, 0.0], [0.0, 0.4]], dtype=DTYPE)
    true_x = tf.constant([4.0, 4.0], dtype=DTYPE)

    # Save config & dedup check
    config = {
        'n_mc': n_mc, 'N': N, 'n_steps': n_steps, 'mu': mu,
        'Q_diff': [4.0, 0.4],
        'true_x': [4.0, 4.0], 'prior_mean': [3.0, 5.0],
        'integrator': 'euler_maruyama',
    }
    path, is_dup = check_and_save(results_dir, config)
    if is_dup:
        print(f"Duplicate run found at {path}, skipping.")
        return

    # Solve BVP for optimal homotopy (offline)
    P_prior_inv = tf.linalg.inv(ssm.P0)
    H_at_prior = ssm.H_jac(ssm.m0)
    R_inv = tf.linalg.inv(ssm.R)
    neg_hess_logh = tf.transpose(H_at_prior) @ R_inv @ H_at_prior

    print("Solving BVP for optimal homotopy...", end=" ", flush=True)
    beta_sched, bdot_sched, lam_grid = solve_optimal_homotopy(
        P_prior_inv, neg_hess_logh, mu=mu, n_grid=n_steps + 1)
    print("done.")
    mid_idx = n_steps // 2
    print(f"  beta*(0.5) = {float(beta_sched[mid_idx]):.4f} (straight: 0.5)")

    # Save homotopy schedule (Figure 2a-c)
    lam_np = lam_grid.numpy()
    beta_np = beta_sched.numpy()
    bdot_np = bdot_sched.numpy()
    # Compute stiffness ratio (Figure 2d)
    bdot_straight = tf.ones_like(bdot_sched)
    r_stiff_straight = compute_stiffness_ratio(
        ssm, P_prior_inv, neg_hess_logh, Q_diff, lam_grid, lam_grid, bdot_straight)
    r_stiff_optimal = compute_stiffness_ratio(
        ssm, P_prior_inv, neg_hess_logh, Q_diff, lam_grid, beta_sched, bdot_sched)

    # Save homotopy + stiffness (Figure 2a-d)
    np.savez(os.path.join(results_dir, 'homotopy.npz'),
             lam=lam_np, beta_star=beta_np, beta_straight=lam_np,
             e=beta_np - lam_np, u_star=bdot_np,
             u_straight=np.ones(len(lam_np)),
             R_stiff_straight=r_stiff_straight,
             R_stiff_optimal=r_stiff_optimal)

    # Monte Carlo runs (Table 1)
    mse_straight = np.zeros(n_mc)
    mse_optimal = np.zeros(n_mc)
    trP_straight = np.zeros(n_mc)
    trP_optimal = np.zeros(n_mc)

    print(f"\nRunning {n_mc} Monte Carlo runs (N={N}, n_steps={n_steps})...")
    print(f"{'MC':>4}  {'MSE_beta':>10}  {'MSE_beta*':>10}  {'tr(P_beta)':>10}  {'tr(P_beta*)':>10}")
    print("-" * 55)

    t_start = time.perf_counter()

    for mc in range(n_mc):
        # Same initial particles for both (CRN)
        rng_init = tf.random.Generator.from_seed(mc)
        L_P0 = tf.linalg.cholesky(ssm.P0)
        noise = rng_init.normal(shape=(N, 2), dtype=DTYPE)
        particles_init = ssm.m0[tf.newaxis, :] + tf.linalg.matvec(L_P0, noise)

        seed_brownian = mc + n_mc

        # Straight-line beta = lam
        flow_straight = StochasticFlow(
            n_particles=N, n_flow_steps=n_steps, Q_diff=Q_diff)
        _, _, x1, P1 = flow_straight.flow_step(
            tf.identity(particles_init), ssm.m0, ssm.P0, ssm, z,
            tf.random.Generator.from_seed(seed_brownian))

        # Optimal beta*
        flow_optimal = StochasticFlow(
            n_particles=N, n_flow_steps=n_steps, Q_diff=Q_diff,
            beta_schedule=beta_sched, beta_dot_schedule=bdot_sched)
        _, _, x2, P2 = flow_optimal.flow_step(
            tf.identity(particles_init), ssm.m0, ssm.P0, ssm, z,
            tf.random.Generator.from_seed(seed_brownian))

        mse_straight[mc] = float(tf.reduce_sum((x1 - true_x) ** 2))
        mse_optimal[mc] = float(tf.reduce_sum((x2 - true_x) ** 2))
        trP_straight[mc] = float(tf.linalg.trace(P1))
        trP_optimal[mc] = float(tf.linalg.trace(P2))

        print(f"{mc+1:4d}  {mse_straight[mc]:10.4f}  {mse_optimal[mc]:10.4f}  "
              f"{trP_straight[mc]:10.2f}  {trP_optimal[mc]:10.2f}")

    elapsed = time.perf_counter() - t_start

    # Save Table 1 results
    np.savez(os.path.join(results_dir, 'mc_results.npz'),
             mse_straight=mse_straight, mse_optimal=mse_optimal,
             trP_straight=trP_straight, trP_optimal=trP_optimal)

    # Print summary
    print("-" * 55)
    print(f"{'avg':>4}  {np.mean(mse_straight):10.4f}  {np.mean(mse_optimal):10.4f}  "
          f"{np.mean(trP_straight):10.2f}  {np.mean(trP_optimal):10.2f}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/n_mc:.1f}s per MC)")
    print(f"\nImprovement ratios:")
    print(f"  MSE:  {np.mean(mse_straight)/np.mean(mse_optimal):.2f}x reduction")
    print(f"  tr(P): {np.mean(trP_straight)/np.mean(trP_optimal):.2f}x reduction")

    print(f"\nPaper (Table 1):")
    print(f"  MSE_beta=13.246  MSE_beta*=9.475  ratio=1.40x")
    print(f"  tr(P_beta)=1535.2  tr(P_beta*)=1028.8  ratio=1.49x")

    print(f"\nResults saved to {results_dir}/")

    return {
        'mse_straight': mse_straight, 'mse_optimal': mse_optimal,
        'trP_straight': trP_straight, 'trP_optimal': trP_optimal,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dai(22) Table 1 Replication')
    parser.add_argument('--n_mc', type=int, default=20)
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--n_steps', type=int, default=29)
    parser.add_argument('--mu', type=float, default=0.2)
    args = parser.parse_args()
    run_experiment(n_mc=args.n_mc, N=args.N, n_steps=args.n_steps, mu=args.mu)
