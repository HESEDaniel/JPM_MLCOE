"""Experiment 2: utility-class comparison with the macro state x_t observable."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow direct execution (`python src/experiments/2_dh_plus_macro_known.py`):
# add the project root (final/) to sys.path so `src.*` resolves. conftest.py
# already handles this for pytest; this block covers the script-execution path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src._results import append_log, get_config_hash, get_run_id, results_dir, set_all_seeds
from src.datasets.dgp_macro import (
    DGPConfig, data_to_choice_dataset, simulate_macro_choice_dgp,
)
from src.choice_models.macro_deephalo import MacroDeepHalo
from src.choice_models.simple_mlp import SimpleMLP
from src.choice_models.simple_mnl import SimpleMNL


def split_indices(B: int, train_frac: float = 0.8, seed: int = 42) -> tuple:
    """Shuffle 0..B-1 and split into train/val index arrays.

    Parameters
    ----------
    B : int
        Number of decisions to split.
    train_frac : float
        Fraction assigned to the training set.
    seed : int
        Seed for the permutation RNG.

    Returns
    -------
    tuple of np.ndarray
        (train_indices, val_indices).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(B)
    n_train = int(train_frac * B)
    return perm[:n_train], perm[n_train:]


def train_and_eval(model, train_ds, val_ds, label, psi_true=None, true_probs_val=None,
                   slate_val=None):
    """Fit a model and score it on the validation split.

    Reports validation NLL, psi recovery against the true loadings (Pearson and
    RMSE) when the model exposes ``psi_offer``, and the RMSE of predicted choice
    probabilities against the ground-truth probabilities when those are given.

    Parameters
    ----------
    model : object
        A choice-learn model exposing fit/evaluate/predict_probas; psi recovery
        is computed only when it carries a non-None ``psi_offer`` attribute.
    train_ds, val_ds : ChoiceDataset
        Training and validation datasets.
    label : str
        Human-readable model name for the result record.
    psi_true : np.ndarray or None
        True per-offer macro loadings; enables psi recovery metrics when given.
    true_probs_val : np.ndarray or None
        Ground-truth choice probabilities on the validation split, shape (B, M).
    slate_val : np.ndarray or None
        Boolean slate indicator on the validation split, shape (B, M); used to
        normalize the probability RMSE by slate size.

    Returns
    -------
    dict
        Result record with label, fitted model, history, losses, psi estimate
        and recovery metrics, probability RMSE, and training time.
    """
    t0 = time.time()
    history = model.fit(train_ds, verbose=0)
    train_time = time.time() - t0

    val_loss = float(model.evaluate(val_ds))

    if psi_true is not None and getattr(model, "psi_offer", None) is not None:
        psi_hat = model.psi_offer.numpy()
        pearson = float(np.corrcoef(psi_true, psi_hat)[0, 1])
        rmse = float(np.sqrt(np.mean((psi_hat - psi_true) ** 2)))
    else:
        psi_hat, pearson, rmse = None, float("nan"), float("nan")

    rmse_true = float("nan")
    if true_probs_val is not None:
        pred = model.predict_probas(val_ds).numpy()
        # RMSE_p: per-decision MSE over in-slate items, normalized by slate size
        # (off-slate probs are 0 for both pred and truth), then averaged and rooted.
        sse = ((pred - true_probs_val) ** 2).sum(axis=1)
        rmse_true = float(np.sqrt(np.mean(sse / slate_val.sum(axis=1))))

    return {
        "label": label, "model": model, "history": history,
        "val_loss": val_loss, "train_loss": float(history["train_loss"][-1]),
        "psi_hat": psi_hat, "pearson": pearson, "rmse": rmse,
        "rmse_true_probs": rmse_true,
        "train_time_sec": train_time,
    }


def run_compare(
    scenario: str = "B",
    seed: int = 42,
    T: int = 200,
    N_t: int = 50,
    K_min: int = 2,
    K_max: int = 5,
    beta_decoy: float | None = None,
    beta_similar: float | None = None,
    epochs: int = 50,
    batch_size: int = 512,
):
    """Simulate one DGP, train all models, and print/return their metrics.

    Parameters
    ----------
    scenario : str
        DGP scenario label passed to the simulator; pass None implicitly via a
        beta override to rebuild without a scenario preset.
    seed : int
        Seed for the DGP and the train/val split.
    T, N_t : int
        Number of macro periods and decisions per period.
    K_min, K_max : int
        Minimum and maximum slate size.
    beta_decoy, beta_similar : float or None
        Optional overrides for the decoy/similarity coefficients; when given,
        the DGP is rebuilt with the scenario preset replaced by these values.
    epochs : int
        Training epochs for every model.
    batch_size : int
        Mini-batch size for every model.

    Returns
    -------
    tuple
        (data, results) where data is the simulated DGP dict and results is the
        list of per-model records from train_and_eval.
    """
    set_all_seeds(seed)

    # 1. DGP - beta_* override if given
    cfg = DGPConfig(T=T, N_t=N_t, K_min=K_min, K_max=K_max)
    data = simulate_macro_choice_dgp(scenario=scenario, cfg=cfg, seed=seed)
    if beta_decoy is not None or beta_similar is not None:
        # rebuild DGP with overridden beta (after scenario applied)
        if beta_decoy is not None:
            cfg.beta_decoy = beta_decoy
        if beta_similar is not None:
            cfg.beta_similar = beta_similar
        data = simulate_macro_choice_dgp(scenario=None, cfg=cfg, seed=seed)
    M = data["M"]
    psi_true = data["_psi_offer"]
    true_probs_all = data["_true_probs"]   # (T, N_t, M)
    B = T * N_t

    # 2. Train/val split
    train_idx, val_idx = split_indices(B, train_frac=0.8, seed=seed)
    true_probs_val = true_probs_all.reshape(B, M)[val_idx]
    slate_val = data["slate_indicator"].reshape(B, M)[val_idx].astype(bool)

    train_ds = data_to_choice_dataset(data, indices=train_idx)
    val_ds = data_to_choice_dataset(data, indices=val_idx)

    # 3. Models
    configs = [
        ("Featureless MacroDeepHalo",
         MacroDeepHalo(M=M, epochs=epochs, batch_size=batch_size, lr=1e-3, optimizer="adam"),
         train_ds, val_ds),
        ("SimpleMNL",
         SimpleMNL(M=M, epochs=epochs, batch_size=batch_size, lr=1e-3, optimizer="adam"),
         train_ds, val_ds),
        ("SimpleMLP",
         SimpleMLP(M=M, epochs=epochs, batch_size=batch_size, lr=1e-3, optimizer="adam"),
         train_ds, val_ds),
    ]

    print(f"=== 2_compare.py (DGP-{scenario}, beta_decoy={cfg.beta_decoy}, beta_sim={cfg.beta_similar}, "
          f"K={K_min}..{K_max}, T={T}, N_t={N_t}, seed={seed}) ===\n")
    results = []
    for label, model, tr, vl in configs:
        r = train_and_eval(model, tr, vl, label, psi_true=psi_true,
                           true_probs_val=true_probs_val, slate_val=slate_val)
        results.append(r)
        print(
            f"  {label:30s}  psiPearson={r['pearson']:.4f}  psiRMSE={r['rmse']:.4f}  "
            f"val_NLL={r['val_loss']:.4f}  RMSEprobs={r['rmse_true_probs']:.4f}  "
            f"time={r['train_time_sec']:.1f}s"
        )

    return data, results


def plot_psi_scatter_multiseed(psi_true_per_seed, psi_hat_per_model, scenario, out_dir):
    """Scatter the all-(seed, j) cloud of psi_hat_j vs psi_j, overlaid per model.

    Each seed of the DGP draws an independent psi_offer, so reducing across
    seeds by averaging is meaningless. We instead plot every (psi_true, psi_hat)
    pair and report the per-seed Pearson distribution in the legend.

    Parameters
    ----------
    psi_true_per_seed : np.ndarray
        True per-offer loadings stacked across seeds, shape (n_seeds, M).
    psi_hat_per_model : dict[str, np.ndarray]
        Estimated loadings per model, each value shaped (n_seeds, M).
    scenario : str
        DGP scenario label, used in the title and output filename.
    out_dir : pathlib.Path
        Directory to write the PNG into (created if missing).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    markers = ["o", "^", "s", "D", "P"]
    colors = ["C0", "C1", "C2", "C3", "C4"]

    fig, ax = plt.subplots(figsize=(7, 5.8))
    for (label, psi_arr), mk, c in zip(psi_hat_per_model.items(), markers, colors):
        n_seeds = psi_arr.shape[0]
        pearsons = np.array([
            np.corrcoef(psi_true_per_seed[s], psi_arr[s])[0, 1]
            for s in range(n_seeds)
        ])
        ax.scatter(psi_true_per_seed.flatten(), psi_arr.flatten(),
                   c=c, marker=mk, s=22, alpha=0.40, edgecolors="none",
                   label=f"{label} ($\\bar r$ = {pearsons.mean():.3f} $\\pm$ {pearsons.std():.3f})")
    all_true = psi_true_per_seed.flatten()
    all_hat = np.concatenate([v.flatten() for v in psi_hat_per_model.values()])
    lo = float(min(all_true.min(), all_hat.min())) - 0.1
    hi = float(max(all_true.max(), all_hat.max())) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\psi_j$ (true; per seed)")
    ax.set_ylabel(r"$\hat{\psi}_j$ (estimated; per seed)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"$\\psi$ recovery (DGP-{scenario}, "
                 f"{psi_arr.shape[0]} seeds $\\times$ {psi_arr.shape[1]} offers)")
    fig.tight_layout()
    fig.savefig(out_dir / f"psi_scatter_{scenario}.png", dpi=160)
    plt.close(fig)
    print(f"  Plot -> {out_dir}/psi_scatter_{scenario}.png")


def main():
    """Run the comparison over multiple seeds, log rows, and write the scatter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="B", choices=list("ABCD"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--N_t", type=int, default=50)
    parser.add_argument("--K_min", type=int, default=2)
    parser.add_argument("--K_max", type=int, default=5)
    parser.add_argument("--beta_decoy", type=float, default=None,
                        help="override beta_decoy (else scenario default)")
    parser.add_argument("--beta_similar", type=float, default=None)
    parser.add_argument("--seeds", default="32,33,34,35,36,37,38,39,40,41")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    config_meta = dict(scenario=args.scenario, T=args.T, N_t=args.N_t,
                       K_min=args.K_min, K_max=args.K_max,
                       beta_decoy=args.beta_decoy, beta_similar=args.beta_similar,
                       epochs=args.epochs)
    config_hash = get_config_hash(config_meta)
    run_id = get_run_id()
    out = results_dir("2_dh_plus_macro_known")

    log_fields = ["run_id", "config_hash", "scenario", "T", "N_t", "K_min", "K_max",
                  "beta_decoy", "beta_similar", "seed", "epochs",
                  "model", "psi_pearson", "psi_rmse", "val_loss", "rmse_true_probs",
                  "train_time_sec"]

    all_rows = []
    psi_hat_per_model: dict[str, list[np.ndarray]] = {}
    psi_true_list: list[np.ndarray] = []
    for seed in seeds:
        print(f"\n=== seed={seed} ===")
        data, results = run_compare(
            scenario=args.scenario, seed=seed,
            T=args.T, N_t=args.N_t, K_min=args.K_min, K_max=args.K_max,
            beta_decoy=args.beta_decoy, beta_similar=args.beta_similar,
            epochs=args.epochs,
        )
        psi_true_list.append(data["_psi_offer"])
        for r in results:
            row = {**config_meta, "seed": seed,
                   "run_id": run_id, "config_hash": config_hash,
                   "model": r["label"], "psi_pearson": r["pearson"],
                   "psi_rmse": r["rmse"], "val_loss": r["val_loss"],
                   "rmse_true_probs": r["rmse_true_probs"],
                   "train_time_sec": r["train_time_sec"]}
            all_rows.append(row)
            append_log(out / "log.csv", row=row, fields=log_fields)
            if r["psi_hat"] is not None:
                psi_hat_per_model.setdefault(r["label"], []).append(r["psi_hat"])

    # Multi-seed psi scatter - all (seed, j) cloud + per-seed Pearson distribution
    psi_true_per_seed = np.stack(psi_true_list, axis=0)               # (n_seeds, M)
    psi_stack = {label: np.stack(arrs, axis=0)
                 for label, arrs in psi_hat_per_model.items()}
    plot_psi_scatter_multiseed(psi_true_per_seed, psi_stack,
                               args.scenario, out / "figures" / run_id)

    # ===== Multi-seed summary per model =====
    print(f"\n=== Summary across {len(seeds)} seeds (DGP-{args.scenario}) ===")
    print(f"{'Model':30s}  {'psi Pearson':>16}  {'val_loss':>16}  {'RMSE_probs':>16}")
    model_labels = [r["label"] for r in results]   # order from last run
    for label in model_labels:
        sub = [r for r in all_rows if r["model"] == label]
        pe = np.array([r["psi_pearson"] for r in sub])
        vl = np.array([r["val_loss"] for r in sub])
        rp = np.array([r["rmse_true_probs"] for r in sub])
        print(f"{label:30s}  {pe.mean():.4f}+/-{pe.std():.4f}  "
              f"{vl.mean():.4f}+/-{vl.std():.4f}  {rp.mean():.4f}+/-{rp.std():.4f}")
    print(f"\nLogged to {out}/log.csv (run_id={run_id})")


if __name__ == "__main__":
    main()
