"""Part 2-1b: PFPF comparison on multi-target acoustic SSM.

Usage:
    cd Q2 && python experiments/part2/exp_1b_multi_acoustic.py
    cd Q2 && python experiments/part2/exp_1b_multi_acoustic.py --quick
    cd Q2 && python experiments/part2/exp_1b_multi_acoustic.py --N 500 --T 40 --nf 29 --n_traj 5 --n_trials 10
"""
import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ssm.multi_target_acoustic import (
    MultiTargetAcousticModel, sample_initial_distribution, compute_omat_trajectory)
from experiments.part2.exp_1b_common import (
    build_configs, run_experiment, exp_lambda_schedule, DTYPE)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'part2', 'exp_1b_multi_acoustic')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--T', type=int, default=None)
    parser.add_argument('--nf', type=int, default=None)
    parser.add_argument('--n_traj', type=int, default=None)
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.02,
                        help='Q_diff = alpha * diag(P0) for Dai22 SDE standalone.')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        defaults = {'N': 100, 'T': 10, 'nf': 10, 'n_traj': 1, 'n_trials': 2}
    else:
        defaults = {'N': 500, 'T': 40, 'nf': 29, 'n_traj': 3, 'n_trials': 5}

    N = args.N or defaults['N']
    T = args.T or defaults['T']
    nf = args.nf or defaults['nf']
    n_traj = args.n_traj or defaults['n_traj']
    n_trials = args.n_trials or defaults['n_trials']

    cfg = {'N': N, 'T': T, 'nf': nf, 'n_traj': n_traj,
           'n_trials': n_trials, 'seed': args.seed,
           'alpha': args.alpha}

    model = MultiTargetAcousticModel()
    lam = exp_lambda_schedule(nf)
    Q_diff = args.alpha * tf.linalg.diag(tf.linalg.diag_part(model.P0)) if args.alpha else None
    configs = build_configs(N, nf, lam=lam, Q_diff=Q_diff)

    def simulate(ssm, rng_np):
        xs, ys = ssm.simulate(T, rng_np)
        m0, P0 = sample_initial_distribution(
            rng_np, ssm.n_targets, ssm.area_size, x0_true=xs[0])
        ssm.m0 = tf.constant(m0, dtype=DTYPE)
        ssm.P0 = tf.constant(P0, dtype=DTYPE)
        return xs, ys

    def metric_ts(xs, res):
        return compute_omat_trajectory(xs, res.m_filt.numpy(), model.n_targets)

    run_experiment(
        f"MultiAcoustic (N={N}, T={T}, nf={nf})",
        model, configs, n_traj, n_trials,
        simulate, metric_ts, "OMAT", OUT_DIR, args.seed,
        config_dict=cfg)


if __name__ == '__main__':
    main()
