"""Experiment 3b: particle-filter state recovery when the latent process is not AR(1)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import time

import numpy as np

from src.datasets.dgp_macro import DGPConfig, data_to_choice_dataset, simulate_macro_choice_dgp
from src.datasets.latent_processes import generate, fit_best_ar1, PROCESSES
from src._results import append_log, get_run_id, results_dir, set_all_seeds
from src.inference import run_pf_q2
from src.choice_models.macro_deephalo import MacroDeepHalo


def metrics(res, x_true, psi_hat, psi_true):
    """Compute state-recovery and cross-section recovery metrics for one run.

    Parameters
    ----------
    res : dict
        PF output with ``x_hat``, ``x_std`` and ``ess`` arrays.
    x_true : np.ndarray
        True latent state path.
    psi_hat : np.ndarray
        Estimated psi offer values.
    psi_true : np.ndarray
        True psi offer values.

    Returns
    -------
    dict
        RMSE, cov95, mean ESS and psi-Pearson correlation.
    """
    err = res["x_hat"] - x_true
    return {
        "rmse":     float(np.sqrt(np.mean(err ** 2))),
        # cov95 uses a Gaussian +/-1.96sigma interval; kept in the log but not
        # reported, since the framework makes no Gaussian-posterior assumption.
        "cov95":    float(np.mean(np.abs(err) <= 1.96 * res["x_std"])),
        "ess_mean": float(np.mean(res["ess"])),
        "psi_pearson": float(np.corrcoef(psi_hat, psi_true)[0, 1]),
        "psi_rmse": float(np.sqrt(np.mean((psi_hat - psi_true) ** 2))),
    }


def main():
    """Run the form-misspecification PF sweep over processes and seeds.

    For each process and seed, simulates the choice DGP with an overridden true
    x_t path, fits Stage-2 on the true x_t, runs the AR(1) PF with the best-fit
    AR(1) params, and logs the recovery metrics to ``log.csv``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="32,33,34,35,36,37,38,39,40,41")
    parser.add_argument("--processes", default="ar1,ar2",
                        help="comma-separated subset of " + ",".join(PROCESSES))
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--N_t", type=int, default=50)
    parser.add_argument("--K_min", type=int, default=2)
    parser.add_argument("--K_max", type=int, default=5)
    parser.add_argument("--epochs_dh", type=int, default=50)
    parser.add_argument("--n_particles", type=int, default=500)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    processes = [p.strip() for p in args.processes.split(",") if p.strip()]
    run_id = get_run_id()
    out_dir = results_dir("3b_pf_form_misspec")

    print(f"=== form-misspec PF  run_id={run_id} ===")
    print(f"  T={args.T} N_t={args.N_t} N={args.n_particles}  processes={processes}  seeds={seeds}")

    fields = ["run_id", "process", "N_t", "seed", "mu_fit", "phi_fit", "sigma_fit",
              "rmse", "cov95", "ess_mean", "psi_pearson", "psi_rmse", "time_sec"]
    all_rows = []
    for process in processes:
        print(f"\n--- process = {process} ---")
        rows = []
        for seed in seeds:
            set_all_seeds(seed)
            rng = np.random.default_rng(seed + 7000)   # x_t-process RNG, decoupled from DGP seed
            x_true = generate(process, args.T, rng)
            data = simulate_macro_choice_dgp(
                scenario="B",
                cfg=DGPConfig(T=args.T, N_t=args.N_t, K_min=args.K_min, K_max=args.K_max),
                seed=seed,
                x_true_override=x_true,
            )
            psi_true = data["_psi_offer"]
            ds = data_to_choice_dataset(data)   # Stage-2 on the (true) x_t
            m = MacroDeepHalo(M=data["M"], epochs=args.epochs_dh,
                              batch_size=512, lr=1e-3, optimizer="adam")
            m.fit(ds, verbose=0)

            mu_f, phi_f, sig_f = fit_best_ar1(x_true)   # filter's best AR(1) for this series
            t0 = time.time()
            res = run_pf_q2(m, data, mu=mu_f, phi=phi_f, sigma=sig_f,
                            n_particles=args.n_particles, seed=seed)
            row = {
                "run_id": run_id, "process": process, "N_t": args.N_t, "seed": seed,
                "mu_fit": round(mu_f, 4), "phi_fit": round(phi_f, 4), "sigma_fit": round(sig_f, 4),
                **metrics(res, x_true, m.psi_offer.numpy(), psi_true),
                "time_sec": time.time() - t0,
            }
            rows.append(row); all_rows.append(row)
            print(f"  seed={seed}  AR1fit(phi={phi_f:+.2f},sig={sig_f:.2f})  "
                  f"RMSE={row['rmse']:.3f} cov95={row['cov95']:.3f} "
                  f"ESS={row['ess_mean']:.0f} psi_r={row['psi_pearson']:+.3f}")
            append_log(out_dir / "log.csv", row=row, fields=fields)

        def agg(key):
            v = np.array([r[key] for r in rows]); return v.mean(), v.std()
        print(f"  [{process}]  RMSE={agg('rmse')[0]:.3f}+/-{agg('rmse')[1]:.3f}  "
              f"cov95={agg('cov95')[0]:.3f}  "
              f"psi_r={agg('psi_pearson')[0]:+.3f}")

    print(f"\n  Logged to {out_dir}/log.csv")


if __name__ == "__main__":
    main()
