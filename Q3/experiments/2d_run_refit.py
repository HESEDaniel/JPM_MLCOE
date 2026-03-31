"""Part 2d: DeepHalo+xi vs Lu25 on context-effect DGP.

Compares eta recovery on the SAME markets (fair comparison, no data split):
  DH:          DeepHalo NN MLE + MCMC on same data
  Lu25:        Joint MCMC on same data (linear, can't capture context)
  DH(NN only): DeepHalo NN prediction only, no MCMC
  Linear:      OLS + market fixed effects

DGP has context effects alpha*(p_j - p_bar)^2 that DeepHalo captures but Lu25 cannot.
Both models use identical data -- the only difference is architecture.

Usage:
    python 2d_run_refit.py --quick                    # smoke test (3 seeds)
    python 2d_run_refit.py --n_seeds 20               # full run
    python 2d_run_refit.py --n_seeds 20 --gpu 0       # specify GPU
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Parse --gpu BEFORE importing TF so CUDA_VISIBLE_DEVICES takes effect
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=int, default=0)
_args, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf

from choice_learn.datasets.dgp import simulate_deephalo_shrinkage_dgp
from choice_learn.models.deep_halo_combined import BayesianDeepHalo, _logit_shares
from choice_learn.models.lu25_bayesian import Lu25Model


RESULTS_DIR = Path(__file__).parent / "results" / "2d_refit"


def compute_eta_metrics(name, samples, true_params):
    """Compute eta/gamma/phi recovery metrics."""
    eta_true = true_params["eta"]
    gamma_true = true_params["gamma"]

    eta_hat = np.mean(samples["eta"], axis=0)
    eta_bias = float(np.mean(np.abs(eta_hat - eta_true)))
    eta_rmse = float(np.sqrt(np.mean((eta_hat - eta_true) ** 2)))
    eta_corr = float(np.corrcoef(eta_hat.ravel(), eta_true.ravel())[0, 1])

    gamma_prob = np.mean(samples["gamma"], axis=0)
    gamma_hat = (gamma_prob > 0.5).astype(float)
    active = gamma_true.astype(bool)
    inactive = ~active

    tpr = float(np.mean(gamma_hat[active])) if active.any() else np.nan
    fpr = float(np.mean(gamma_hat[inactive])) if inactive.any() else np.nan
    phi_hat = float(np.mean(samples["phi"]))

    return {
        f"{name}_eta_bias": eta_bias,
        f"{name}_eta_rmse": eta_rmse,
        f"{name}_eta_corr": eta_corr,
        f"{name}_gamma_tpr": tpr,
        f"{name}_gamma_fpr": fpr,
        f"{name}_phi_mean": phi_hat,
    }


def compute_share_rmse(shares_pred, shares_true):
    """RMSE between predicted and true shares (including outside option)."""
    return float(np.sqrt(np.mean((shares_pred - shares_true) ** 2)))


def run_single(seed, T, J, N, dgp_type, alpha,
               context_type, nn_epochs, n_mcmc, n_burnin, chain_dir=None):
    """Run all models on one dataset (same data, no split), return metrics."""
    X, q, true_params = simulate_deephalo_shrinkage_dgp(
        T=T, J=J, N=N, dgp_type=dgp_type, alpha=alpha,
        context_type=context_type, seed=seed)

    true_eta_gamma = {
        "eta": true_params["eta"],
        "gamma": true_params["gamma"],
    }
    shares_true = true_params["shares"]

    row = {"seed": seed, "dgp": dgp_type, "alpha": alpha,
           "context_type": context_type, "T": T, "J": J, "N": N}

    print(f"\n  --- BDH (seed={seed}) ---")
    t0 = time.time()
    dh = BayesianDeepHalo(
        embedding_dim=16, n_layers=2, n_heads=2,
        nn_lr=1e-3, nn_epochs=nn_epochs,
        n_mcmc=n_mcmc, n_burnin=n_burnin,
    )
    dh.fit(X, q)
    row["dh_elapsed_sec"] = round(time.time() - t0, 2)

    dh_shares = dh.compute_shares(X)
    dh_metrics = compute_eta_metrics("dh", dh.samples_, true_eta_gamma)
    row.update(dh_metrics)
    row["dh_share_rmse"] = compute_share_rmse(dh_shares, shares_true)

    print(f"  --- DH(NN only) (seed={seed}) ---")
    nn_shares = _logit_shares(dh._h_fixed).numpy()
    row["nn_share_rmse"] = compute_share_rmse(nn_shares, shares_true)

    print(f"  --- Lu25 (seed={seed}) ---")
    t0 = time.time()
    lu = Lu25Model(
        rc_indices=[],
        R0=1,
        n_mcmc=n_mcmc, n_burnin=n_burnin,
        use_tmh=False,
    )
    lu.fit(X, q)
    row["lu_elapsed_sec"] = round(time.time() - t0, 2)

    lu_metrics = compute_eta_metrics("lu", lu.samples_, true_eta_gamma)
    row.update(lu_metrics)

    # Lu25 predicted shares
    lu_eta = np.mean(lu.samples_["eta"], axis=0)
    lu_xi = np.mean(lu.samples_["xi_bar"], axis=0)
    lu_beta = np.mean(lu.samples_["beta_bar"], axis=0)
    lu_v = (np.sum(X * lu_beta[np.newaxis, np.newaxis, :], axis=-1)
            + lu_xi[:, np.newaxis] + lu_eta)
    lu_v_all = np.concatenate([np.zeros((T, 1)), lu_v], axis=1)
    lu_shares = np.exp(lu_v_all - lu_v_all.max(axis=1, keepdims=True))
    lu_shares = lu_shares / lu_shares.sum(axis=1, keepdims=True)
    row["lu_share_rmse"] = compute_share_rmse(lu_shares, shares_true)

    print(f"  --- Linear (seed={seed}) ---")
    s_obs = q / np.sum(q, axis=1, keepdims=True)
    s_obs = np.clip(s_obs, 1e-10, 1.0)
    log_ratio = np.log(s_obs[:, 1:]) - np.log(s_obs[:, 0:1])
    y = log_ratio.ravel()
    X_ols = np.concatenate([
        X.reshape(-1, 2),
        np.repeat(np.eye(T), J, axis=0),
    ], axis=1)
    beta_ols = np.linalg.lstsq(X_ols, y, rcond=None)[0]
    y_hat = X_ols @ beta_ols
    v_lin = y_hat.reshape(T, J)
    v_lin_all = np.concatenate([np.zeros((T, 1)), v_lin], axis=1)
    lin_shares = np.exp(v_lin_all - v_lin_all.max(axis=1, keepdims=True))
    lin_shares = lin_shares / lin_shares.sum(axis=1, keepdims=True)
    row["lin_share_rmse"] = compute_share_rmse(lin_shares, shares_true)

    if chain_dir is not None:
        np.savez_compressed(
            chain_dir / f"dh_seed{seed}_{dgp_type}_a{alpha}.npz",
            eta=dh.samples_["eta"], gamma=dh.samples_["gamma"],
            phi=dh.samples_["phi"], h_nn=dh._h_fixed.numpy(),
            full_eta=dh.full_chain_[0],
            full_gamma=dh.full_chain_[1],
            full_phi=dh.full_chain_[2])
        np.savez_compressed(
            chain_dir / f"lu_seed{seed}_{dgp_type}_a{alpha}.npz",
            eta=lu.samples_["eta"], gamma=lu.samples_["gamma"],
            phi=lu.samples_["phi"], xi_bar=lu.samples_["xi_bar"],
            beta_bar=lu.samples_["beta_bar"],
            full_beta=lu.full_chain_[0],
            full_r=lu.full_chain_[1],
            full_xi=lu.full_chain_[2],
            full_eta=lu.full_chain_[3],
            full_gamma=lu.full_chain_[4],
            full_phi=lu.full_chain_[5])

    print(f"\n  [seed={seed}, {dgp_type}, alpha={alpha}]")
    print(f"    BDH:         eta_bias={dh_metrics['dh_eta_bias']:.3f} "
          f"eta_corr={dh_metrics['dh_eta_corr']:.3f} "
          f"TPR={dh_metrics['dh_gamma_tpr']:.3f} "
          f"FPR={dh_metrics['dh_gamma_fpr']:.3f} "
          f"RMSE={row['dh_share_rmse']:.5f}")
    print(f"    Lu25:        eta_bias={lu_metrics['lu_eta_bias']:.3f} "
          f"eta_corr={lu_metrics['lu_eta_corr']:.3f} "
          f"TPR={lu_metrics['lu_gamma_tpr']:.3f} "
          f"FPR={lu_metrics['lu_gamma_fpr']:.3f} "
          f"RMSE={row['lu_share_rmse']:.5f}")
    print(f"    BDH(NN only): RMSE={row['nn_share_rmse']:.5f}")
    print(f"    Linear:      RMSE={row['lin_share_rmse']:.5f}")

    return row


def print_summary(df, dgp_type, alpha=None):
    """Print summary table for one DGP type and alpha."""
    sub = df[df["dgp"] == dgp_type].dropna(subset=["dh_eta_bias"])
    if alpha is not None:
        sub = sub[sub["alpha"] == alpha]
    if len(sub) == 0:
        return

    print(f"\n{'='*75}")
    print(f"  DGP: {dgp_type}, alpha={alpha} ({len(sub)} seeds)")
    print(f"{'='*75}")

    # eta recovery
    print(f"\n  {'Model':>12} {'eta Bias':>12} {'eta RMSE':>12} "
          f"{'eta Corr':>12} {'TPR(gamma)':>10} {'FPR(gamma)':>10}")
    print(f"  {'-'*70}")
    for prefix, label in [("dh", "BDH"), ("lu", "Lu25")]:
        bias = sub[f"{prefix}_eta_bias"]
        rmse = sub[f"{prefix}_eta_rmse"]
        corr = sub[f"{prefix}_eta_corr"]
        tpr = sub[f"{prefix}_gamma_tpr"]
        fpr = sub[f"{prefix}_gamma_fpr"]
        print(f"  {label:>12} "
              f"{bias.mean():>5.3f}+/-{bias.std():>4.3f} "
              f"{rmse.mean():>5.3f}+/-{rmse.std():>4.3f} "
              f"{corr.mean():>5.3f}+/-{corr.std():>4.3f} "
              f"{tpr.mean():>5.3f}+/-{tpr.std():.3f} "
              f"{fpr.mean():>5.3f}+/-{fpr.std():.3f}")

    # Share RMSE
    print(f"\n  {'Model':>12} {'Share RMSE':>14} {'Shock Gain':>12}")
    print(f"  {'-'*42}")
    nn_rmse = sub["nn_share_rmse"].mean()
    for col, label in [("dh_share_rmse", "BDH"),
                       ("lu_share_rmse", "Lu25"),
                       ("nn_share_rmse", "BDH(NN only)"),
                       ("lin_share_rmse", "Linear")]:
        r = sub[col]
        gain = ""
        if col in ("dh_share_rmse", "lu_share_rmse"):
            g = (nn_rmse - r.mean()) / nn_rmse * 100
            gain = f"{g:>+8.1f}%"
        print(f"  {label:>12} {r.mean():>8.5f}+/-{r.std():.5f} {gain}")

    # Timing
    print(f"\n  Timing:")
    print(f"    DH:   {sub['dh_elapsed_sec'].mean():.1f}s")
    print(f"    Lu25: {sub['lu_elapsed_sec'].mean():.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Part 2d: DeepHalo vs Lu25 on context-effect DGP (same data)")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--J", type=int, default=10)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--nn_epochs", type=int, default=300)
    parser.add_argument("--n_mcmc", type=int, default=5000)
    parser.add_argument("--n_burnin", type=int, default=1500)
    parser.add_argument("--dgp", default="context_sparse",
                        choices=["context_sparse", "context_nonsparse",
                                 "nocontext_sparse", "nocontext_nonsparse"])
    parser.add_argument("--alpha", type=float, nargs="+",
                        default=[-1.0, -3.0],
                        help="Context effect strengths to test")
    parser.add_argument("--context_type", default="quadratic",
                        choices=["quadratic", "rank", "asymmetric"],
                        help="Type of nonlinear context effect")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_seeds = 3
        args.nn_epochs = 50
        args.n_mcmc = 500
        args.n_burnin = 200
        args.alpha = [-1.0]

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_dir = out_dir / "chains"
    chain_dir.mkdir(exist_ok=True)
    config_dir = out_dir / "configs"
    config_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "2d_refit_results.csv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "T": args.T, "J": args.J, "N": args.N,
        "n_seeds": args.n_seeds, "seed_start": args.seed_start,
        "dgp": args.dgp, "alphas": args.alpha,
        "context_type": args.context_type,
        "nn_epochs": args.nn_epochs,
        "n_mcmc": args.n_mcmc, "n_burnin": args.n_burnin,
        "gpu": args.gpu, "timestamp": timestamp,
    }
    with open(config_dir / f"refit_{timestamp}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Part 2d: DeepHalo vs Lu25 (Context-Effect DGP, Same Data)")
    print(f"  n_seeds={args.n_seeds}, seed_start={args.seed_start}")
    print(f"  T={args.T}, J={args.J}, N={args.N}")
    print(f"  DGP: {args.dgp}, alpha={args.alpha}, context_type={args.context_type}")
    print(f"  nn_epochs={args.nn_epochs}, n_mcmc={args.n_mcmc}, n_burnin={args.n_burnin}")
    print(f"  GPU={args.gpu}")

    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    results = []

    for alpha in args.alpha:
        for i in range(args.n_seeds):
            seed = args.seed_start + i
            print(f"\n{'='*75}")
            print(f"  [{i+1}/{args.n_seeds}] seed={seed}, dgp={args.dgp}, "
                  f"alpha={alpha}, context={args.context_type}")
            print(f"{'='*75}")

            try:
                row = run_single(
                    seed=seed, T=args.T,
                    J=args.J, N=args.N, dgp_type=args.dgp, alpha=alpha,
                    context_type=args.context_type,
                    nn_epochs=args.nn_epochs,
                    n_mcmc=args.n_mcmc, n_burnin=args.n_burnin,
                    chain_dir=chain_dir,
                )
                results.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results.append({"seed": seed, "dgp": args.dgp,
                                "alpha": alpha, "error": str(e)})
                continue

            row_df = pd.DataFrame([row])
            row_df.to_csv(csv_path, mode="a", header=not csv_exists, index=False)
            csv_exists = True

    # Summary
    df = pd.DataFrame(results)
    for alpha in args.alpha:
        print_summary(df, args.dgp, alpha)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
