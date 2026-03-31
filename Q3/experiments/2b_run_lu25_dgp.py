"""Run Lu & Shimizu (2025) DGP 1-4 simulation experiments.

Replicates paper Section 4 (Tables 1-2):
  - DGP1: sparse shocks, exogenous price
  - DGP2: sparse shocks, endogenous price
  - DGP3: non-sparse shocks, exogenous price
  - DGP4: non-sparse shocks, endogenous price

Usage:
    python run_lu25_dgp.py                          # all DGPs, default settings
    python run_lu25_dgp.py --dgp dgp1 dgp2          # specific DGPs
    python run_lu25_dgp.py --T 100 --J 15            # larger market
    python run_lu25_dgp.py --n_rep 50 --n_mcmc 10000 # full replication
    python run_lu25_dgp.py --quick                   # quick smoke test
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from choice_learn.models.lu25_bayesian import Lu25Model
from choice_learn.datasets.dgp import simulate_lu25_dgp

RESULTS_DIR = Path(__file__).parent / "results" / "lu25"


def run_single(dgp_type, T, J, N, n_mcmc, n_burnin, R0, seed, chain_dir=None):
    """Run a single DGP experiment and return results dict.

    Saves per-replication arrays (xi_jt, xi_bar_t, true values) to chain_dir if provided.
    """
    X, q, true_params = simulate_lu25_dgp(T=T, J=J, N=N, dgp_type=dgp_type, seed=seed)

    model = Lu25Model(
        rc_indices=[0],  # RC on price only (paper setup)
        R0=R0,
        n_mcmc=n_mcmc,
        n_burnin=n_burnin,
    )

    t0 = time.time()
    model.fit(X, q)
    elapsed = time.time() - t0

    summary = model.get_posterior_summary()

    beta_hat = summary["beta_bar"]["mean"]               # E[beta_bar]
    sigma_hat = np.mean(np.exp(model.samples_["r"]), axis=0)  # E[exp(r)], NOT exp(E[r])

    beta_true = true_params["beta_bar"]
    sigma_true = true_params["sigma"]

    # Per-market intercept xi_bar_t
    xi_bar_hat = summary["xi_bar"]["mean"]  # (T,)

    # xi_jt = xi_bar_t + eta_jt
    eta_hat = np.mean(model.samples_["eta"], axis=0)     # (T, J)
    xi_hat = xi_bar_hat[:, np.newaxis] + eta_hat          # (T, J)
    xi_true = true_params["xi_bar"] + true_params["eta"]  # (T, J)

    # Intercept (xi_bar_t) within-rep summary
    xi_bar_true = true_params["xi_bar"]
    xi_bar_true_arr = (np.full(T, float(xi_bar_true)) if np.ndim(xi_bar_true) == 0
                       else np.asarray(xi_bar_true))
    int_abs_bias = float(np.mean(np.abs(xi_bar_hat - xi_bar_true_arr)))
    int_sd = float(np.std(xi_bar_hat))  # cross-sectional std of xi_bar_hat itself

    # xi_jt within-rep summary
    xi_abs_bias = float(np.mean(np.abs(xi_hat - xi_true)))
    xi_sd = float(np.std(xi_hat))  # cross-sectional std of xi_hat_jt itself (paper style)

    # Prob(gamma=1) by true sparsity -- only for DGP1/2 (paper Table 1)
    true_gamma = true_params["gamma"]
    if true_gamma is not None:
        gamma_post_mean = np.mean(model.samples_["gamma"], axis=0)  # (T, J)
        prob_nonsparse = float(np.mean(gamma_post_mean[true_gamma > 0.5]))
        prob_sparse = float(np.mean(gamma_post_mean[true_gamma < 0.5]))
    else:
        prob_nonsparse = float('nan')
        prob_sparse = float('nan')

    if chain_dir is not None:
        np.savez(
            chain_dir / f"chain_{dgp_type}_seed{seed}.npz",
            # MCMC posterior samples
            beta_bar=model.samples_["beta_bar"],  # (n_samples, K)
            r=model.samples_["r"],                # (n_samples, 1)
            xi_bar=model.samples_["xi_bar"],      # (n_samples, T)
            eta=model.samples_["eta"],            # (n_samples, T, J)
            gamma=model.samples_["gamma"],        # (n_samples, T, J)
            phi=model.samples_["phi"],            # (n_samples, T)
            # Point estimates
            xi_bar_hat=xi_bar_hat,                # (T,)
            xi_hat=xi_hat,                        # (T, J)
            # True values
            beta_true=beta_true,                  # (K,)
            sigma_true=sigma_true,                # (1,)
            xi_bar_true=xi_bar_true_arr,          # (T,)
            xi_true=xi_true,                      # (T, J)
            eta_true=true_params["eta"],          # (T, J)
            gamma_true=true_params["gamma"] if true_params["gamma"] is not None else np.array([]),
        )

    result = {
        "dgp": dgp_type,
        "T": T,
        "J": J,
        "seed": seed,
        "elapsed_sec": round(elapsed, 1),
        "beta_p_true": float(beta_true[0]),
        "beta_p_est": float(beta_hat[0]),
        "beta_w_true": float(beta_true[1]),
        "beta_w_est": float(beta_hat[1]),
        "sigma_true": float(sigma_true[0]),
        "sigma_est": float(sigma_hat[0]),
        "int_abs_bias": int_abs_bias,
        "int_sd": int_sd,
        "xi_jt_abs_bias": xi_abs_bias,
        "xi_jt_sd": xi_sd,
        "prob_nonsparse": prob_nonsparse,
        "prob_sparse": prob_sparse,
    }
    return result


def run_experiments(dgps, T, J, N, n_rep, n_mcmc, n_burnin, R0, output_dir, seed_start=42,
                    method="tmh"):
    """Run all experiments and save results incrementally."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_csv = output_dir / "lu25_results.csv"

    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    config = {
        "dgps": dgps, "T": T, "J": J, "N": N,
        "n_rep": n_rep, "n_mcmc": n_mcmc, "n_burnin": n_burnin, "R0": R0,
        "seed_start": seed_start, "method": method, "timestamp": timestamp,
    }
    config_path = config_dir / f"lu25_T{T}_J{J}_{timestamp}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    master_exists = master_csv.exists() and master_csv.stat().st_size > 0

    all_results = []
    total = len(dgps) * n_rep
    count = 0

    chain_dir = output_dir / "chains"
    chain_dir.mkdir(exist_ok=True)

    for dgp in dgps:
        print(f"\n{'='*60}")
        print(f"  {dgp.upper()} | T={T}, J={J}, N={N}, n_mcmc={n_mcmc}")
        print(f"{'='*60}")

        for rep in range(n_rep):
            count += 1
            seed = seed_start + rep
            print(f"\n[{count}/{total}] {dgp} rep={rep+1}/{n_rep} (seed={seed})")

            try:
                result = run_single(dgp, T, J, N, n_mcmc, n_burnin, R0, seed,
                                    chain_dir=chain_dir)
                result["method"] = method
                all_results.append(result)

                print(
                    f"  beta_p: {result['beta_p_est']:.3f} (true={result['beta_p_true']:.1f}), "
                    f"beta_w: {result['beta_w_est']:.3f} (true={result['beta_w_true']:.1f}), "
                    f"sigma: {result['sigma_est']:.3f} (true={result['sigma_true']:.1f}), "
                    f"Prob: {result['prob_nonsparse']:.2f}/{result['prob_sparse']:.2f}, "
                    f"time: {result['elapsed_sec']:.1f}s"
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                result = {"dgp": dgp, "seed": seed, "method": method,
                          "error": str(e)}
                all_results.append(result)

            row_df = pd.DataFrame([result])
            row_df.to_csv(master_csv, mode='a', header=not master_exists, index=False)
            master_exists = True

    print(f"\nResults saved to {master_csv}")
    print(f"Per-rep arrays saved to {chain_dir}/")

    df = pd.DataFrame(all_results)
    print_summary(df, dgps, chain_dir)

    return df


def load_chain_arrays(chain_dir, dgp, seeds):
    """Load per-rep xi/int arrays from chain NPZ files.

    Returns:
        dict with keys 'xi_hat', 'xi_true', 'int_hat', 'int_true', 'r_samples'
        each as list of arrays, or empty dict if no files found.
    """
    arrays = {"xi_hat": [], "xi_true": [], "int_hat": [], "int_true": [], "r_samples": []}
    for seed in seeds:
        path = chain_dir / f"chain_{dgp}_seed{seed}.npz"
        if path.exists():
            d = np.load(path)
            arrays["xi_hat"].append(d["xi_hat"])
            arrays["xi_true"].append(d["xi_true"])
            arrays["int_hat"].append(d["xi_bar_hat"])
            arrays["int_true"].append(d["xi_bar_true"])
            if "r" in d:
                arrays["r_samples"].append(d["r"])
    return arrays


def compute_paper_metrics(hats, trues):
    """Compute paper-style metrics.

    Bias = mean over reps of [mean over (j,t) of |xi_hat_jt - xi_true_jt|]
    SD   = mean over reps of [std over (j,t) of xi_hat_jt]  (cross-sectional spread of estimates)

    Args:
        hats: np.ndarray (n_rep, ...) -- estimates across reps
        trues: np.ndarray (n_rep, ...) -- true values across reps

    Returns:
        (abs_bias, sd): paper-style metrics
    """
    n_rep = hats.shape[0]
    biases = []
    sds = []
    for r in range(n_rep):
        biases.append(np.mean(np.abs(hats[r] - trues[r])))
        sds.append(np.std(hats[r]))
    return float(np.mean(biases)), float(np.mean(sds))


def print_summary(df, dgps, chain_dir=None):
    """Print paper-style summary table (Table 1/2 format)."""
    print(f"\n{'='*120}")
    print("  SUMMARY (across replications)")
    print(f"{'='*120}")
    print(f"{'DGP':<6} {'beta_p':>10} {'':>8} {'beta_w':>10} {'':>8} "
          f"{'sigma':>10} {'':>8} {'Int(xi_bar)':>10} {'':>8} "
          f"{'xi(xi_jt)':>10} {'':>8} {'P(eta!=0)':>8} {'P(eta=0)':>8}")
    print(f"{'':>6} {'Bias':>10} {'SD':>8} {'Bias':>10} {'SD':>8} "
          f"{'Bias':>10} {'SD':>8} {'|Bias|':>10} {'SD':>8} "
          f"{'|Bias|':>10} {'SD':>8}")
    print("-" * 120)

    for dgp in dgps:
        sub = df[df["dgp"] == dgp].dropna(subset=["beta_p_est"])
        if len(sub) == 0:
            print(f"{dgp:<6} {'(no results)':>10}")
            continue

        bp_bias = sub["beta_p_est"] - sub["beta_p_true"]
        bw_bias = sub["beta_w_est"] - sub["beta_w_true"]
        sg_bias = sub["sigma_est"] - sub["sigma_true"]

        line = (
            f"{dgp:<6} "
            f"{bp_bias.mean():>10.4f} {bp_bias.std():>8.4f} "
            f"{bw_bias.mean():>10.4f} {bw_bias.std():>8.4f} "
            f"{sg_bias.mean():>10.4f} {sg_bias.std():>8.4f} "
        )

        # Load chain arrays for paper-style Int/xi metrics
        int_str = f"{'---':>10} {'---':>8} "
        xi_str = f"{'---':>10} {'---':>8} "

        if chain_dir is not None:
            seeds = sub["seed"].tolist()
            arrays = load_chain_arrays(Path(chain_dir), dgp, seeds)

            if len(arrays["int_hat"]) >= 2:
                int_b, int_s = compute_paper_metrics(
                    np.stack(arrays["int_hat"]), np.stack(arrays["int_true"]))
                int_str = f"{int_b:>10.4f} {int_s:>8.4f} "

            if len(arrays["xi_hat"]) >= 2:
                xi_b, xi_s = compute_paper_metrics(
                    np.stack(arrays["xi_hat"]), np.stack(arrays["xi_true"]))
                xi_str = f"{xi_b:>10.4f} {xi_s:>8.4f} "

        line += int_str + xi_str
        line += f"{sub['prob_nonsparse'].mean():>8.2f} {sub['prob_sparse'].mean():>8.2f}"
        print(line)

    print(f"{'='*120}")
    print("  True: beta_p=-1.0, beta_w=0.5, sigma=1.5, xi_bar=-1.0")
    print("  Int/xi: paper-style per-element across-rep |bias|/SD (need n_rep>=2 and chain files)")


def main():
    parser = argparse.ArgumentParser(description="Run Lu25 DGP simulations")
    parser.add_argument("--dgp", nargs="+", default=["dgp1", "dgp2", "dgp3", "dgp4"],
                        choices=["dgp1", "dgp2", "dgp3", "dgp4"])
    parser.add_argument("--T", type=int, default=25, help="Number of markets")
    parser.add_argument("--J", type=int, default=5, help="Number of products per market")
    parser.add_argument("--N", type=int, default=1000, help="Consumers per market")
    parser.add_argument("--n_rep", type=int, default=1, help="Number of replications")
    parser.add_argument("--n_mcmc", type=int, default=10000, help="MCMC iterations")
    parser.add_argument("--n_burnin", type=int, default=3000, help="Burn-in iterations")
    parser.add_argument("--R0", type=int, default=200, help="MC draws for shares")
    parser.add_argument("--seed_start", type=int, default=42, help="Starting seed")
    parser.add_argument("--method", type=str, default="tmh",
                        choices=["tmh", "rwmh"], help="MCMC method label")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (1 rep, 2000 iter, 500 burnin)")
    args = parser.parse_args()

    if args.quick:
        args.n_rep = 1
        args.n_mcmc = 2000
        args.n_burnin = 500

    print("Lu25 DGP Experiment")
    print(f"  DGPs: {args.dgp}")
    print(f"  T={args.T}, J={args.J}, N={args.N}")
    print(f"  n_rep={args.n_rep}, n_mcmc={args.n_mcmc}, n_burnin={args.n_burnin}, R0={args.R0}")
    print(f"  Output: {args.output_dir}")

    run_experiments(
        dgps=args.dgp,
        T=args.T,
        J=args.J,
        N=args.N,
        n_rep=args.n_rep,
        n_mcmc=args.n_mcmc,
        n_burnin=args.n_burnin,
        R0=args.R0,
        output_dir=args.output_dir,
        seed_start=args.seed_start,
        method=args.method,
    )


if __name__ == "__main__":
    main()
