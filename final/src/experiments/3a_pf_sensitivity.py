"""Experiment 3a: particle-filter state recovery under state-equation parameter perturbations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import time

import numpy as np

from src.datasets.dgp_macro import DGPConfig, data_to_choice_dataset, simulate_macro_choice_dgp
from src._results import append_log, get_run_id, results_dir, set_all_seeds
from src.inference import run_pf_q2
from src.choice_models.macro_deephalo import MacroDeepHalo


MU_TRUE, PHI_TRUE, SIG_TRUE = 0.0, 0.9, 0.5

# First row is the oracle baseline; the rest perturb one parameter at a time.
CONFIGS = [
    ("oracle",     MU_TRUE,  PHI_TRUE, SIG_TRUE),
    ("mu_plus",     0.2,     PHI_TRUE, SIG_TRUE),
    ("mu_minus",   -0.2,     PHI_TRUE, SIG_TRUE),
    ("phi_stress", MU_TRUE,  0.5,      SIG_TRUE),   # severe phi misspec for systematic vs noise test
    ("phi_under",  MU_TRUE,  0.7,      SIG_TRUE),
    ("phi_over",   MU_TRUE,  0.99,     SIG_TRUE),
    ("sig_under",  MU_TRUE, PHI_TRUE,  0.2),
    ("sig_over",   MU_TRUE, PHI_TRUE,  1.0),
]


def metrics(res, x_true):
    """Compute state-recovery metrics for a particle-filter result.

    Parameters
    ----------
    res : dict
        PF output with keys ``x_hat`` (posterior mean), ``x_std`` (posterior
        std), and ``ess`` (effective sample size per step).
    x_true : array-like
        Ground-truth latent state trajectory.

    Returns
    -------
    dict
        Metrics ``rmse``, ``cov95`` (95% interval coverage), ``ess_mean``, and
        ``ess_min``.
    """
    err = res["x_hat"] - x_true
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "cov95": float(np.mean(np.abs(err) <= 1.96 * res["x_std"])),
        "ess_mean": float(np.mean(res["ess"])),
        "ess_min":  float(np.min(res["ess"])),
    }


def main():
    """Run the PF SSM-sensitivity sweep over seeds and log per-config metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="32,33,34,35,36,37,38,39,40,41")
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--N_t", type=int, default=50)
    parser.add_argument("--K_min", type=int, default=2)
    parser.add_argument("--K_max", type=int, default=5)
    parser.add_argument("--epochs_dh", type=int, default=50)
    parser.add_argument("--n_particles", type=int, default=500)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    run_id = get_run_id()
    out_dir = results_dir("3a_pf_sensitivity")

    print(f"=== PF SSM-sensitivity  run_id={run_id} ===")
    print(f"  T={args.T} N_t={args.N_t} N={args.n_particles}  seeds={seeds}")
    print(f"  truth (mu, phi, sigma) = ({MU_TRUE}, {PHI_TRUE}, {SIG_TRUE})")

    fields = ["run_id", "seed", "config", "mu", "phi", "sigma",
              "rmse", "cov95", "ess_mean", "ess_min", "time_sec"]
    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        data = simulate_macro_choice_dgp(
            scenario="B",
            cfg=DGPConfig(T=args.T, N_t=args.N_t, K_min=args.K_min, K_max=args.K_max),
            seed=seed,
        )
        x_true = data["_x_true"]
        ds = data_to_choice_dataset(data)
        m = MacroDeepHalo(M=data["M"], epochs=args.epochs_dh,
                          batch_size=512, lr=1e-3, optimizer="adam")
        m.fit(ds, verbose=0)
        print(f"\n  seed={seed}")
        for label, mu, phi, sig in CONFIGS:
            t0 = time.time()
            res = run_pf_q2(m, data, mu=mu, phi=phi, sigma=sig,
                             n_particles=args.n_particles, seed=seed)
            row = {
                "run_id": run_id, "seed": seed, "config": label,
                "mu": mu, "phi": phi, "sigma": sig,
                **metrics(res, x_true),
                "time_sec": time.time() - t0,
            }
            rows.append(row)
            print(f"    {label:<10}  RMSE={row['rmse']:.3f}  cov95={row['cov95']:.3f}  "
                  f"ESS={row['ess_mean']:.0f}")
            append_log(out_dir / "log.csv", row=row, fields=fields)

    print("\n=== summary (mean +/- std over seeds) ===")
    print(f"  {'config':<10}  {'RMSE':>13}  {'cov95':>13}  {'ESS':>13}")
    for label, _, _, _ in CONFIGS:
        seed_rows = [r for r in rows if r["config"] == label]
        rmse = np.array([r["rmse"] for r in seed_rows])
        cov = np.array([r["cov95"] for r in seed_rows])
        ess = np.array([r["ess_mean"] for r in seed_rows])
        print(f"  {label:<10}  {rmse.mean():.3f}+/-{rmse.std():.3f}  "
              f"{cov.mean():.3f}+/-{cov.std():.3f}  {ess.mean():.0f}+/-{ess.std():.0f}")
    print(f"\n  Logged to {out_dir}/log.csv")


if __name__ == "__main__":
    main()
