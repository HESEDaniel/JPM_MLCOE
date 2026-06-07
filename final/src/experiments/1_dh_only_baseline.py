"""Experiment 1: macro-blind DeepHalo on a macro-aware DGP, the paired baseline for Experiment 2."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import time

import numpy as np

from choice_learn.tf_ops import CustomCategoricalCrossEntropy

from src._results import append_log, get_config_hash, get_run_id, results_dir, set_all_seeds
from src.choice_models.deephalo import FeaturelessDeepHalo
from src.datasets.dgp_macro import DGPConfig, data_to_choice_dataset, simulate_macro_choice_dgp


def split_indices(B: int, train_frac: float = 0.8, seed: int = 42):
    """Split B decisions into a shuffled train/validation index partition.

    Parameters
    ----------
    B : int
        Total number of decisions to split.
    train_frac : float, optional
        Fraction of decisions assigned to the training set.
    seed : int, optional
        Seed for the permutation RNG.

    Returns
    -------
    tuple of numpy.ndarray
        Training indices and validation indices.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(B)
    n_train = int(train_frac * B)
    return perm[:n_train], perm[n_train:]


def run_one(seed: int, T: int, N_t: int, K_min: int, K_max: int,
            epochs: int, batch_size: int) -> dict:
    """Train a macro-blind FeaturelessDeepHalo on one DGP-B draw and score it.

    Parameters
    ----------
    seed : int
        Seed controlling the DGP draw and all model randomness.
    T : int
        Number of time periods in the DGP.
    N_t : int
        Number of decisions per period.
    K_min, K_max : int
        Minimum and maximum slate size in the DGP.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.

    Returns
    -------
    dict
        Seed, validation loss, final training loss, RMSE of predicted probs
        against the ground-truth softmax probs, and training time in seconds.
    """
    set_all_seeds(seed)
    data = simulate_macro_choice_dgp(
        scenario="B",
        cfg=DGPConfig(T=T, N_t=N_t, K_min=K_min, K_max=K_max),
        seed=seed,
    )
    M = data["M"]
    B = T * N_t
    train_idx, val_idx = split_indices(B, train_frac=0.8, seed=seed)

    # DH-only training: no shared_features (so the macro term never enters).
    train_ds = data_to_choice_dataset(data, indices=train_idx)
    val_ds = data_to_choice_dataset(data, indices=val_idx)

    model = FeaturelessDeepHalo(
        n_items=M, n_layers=3, init="he",
        optimizer="adam", lr=1e-3, epochs=epochs, batch_size=batch_size,
    )
    # Train with the NLL (categorical cross-entropy), matching Experiment 2, so the
    # paired gap isolates the macro term and not the training loss. The
    # FeaturelessDeepHalo default is the paper's MSE (Q3 replication only).
    model.loss = CustomCategoricalCrossEntropy()
    t0 = time.time()
    history = model.fit(train_ds, verbose=0)
    train_time = time.time() - t0
    val_loss = float(model.evaluate(val_ds))

    # RMSE against truth: compare predicted softmax probs to the
    # ground-truth softmax(V_true) saved in the DGP.
    true_probs_val = data["_true_probs"].reshape(B, M)[val_idx]
    slate_val = data["slate_indicator"].reshape(B, M)[val_idx].astype(bool)
    pred = model.predict_probas(val_ds).numpy()
    # RMSE_p: per-decision MSE over the in-slate items (off-slate probs are 0 for
    # both prediction and truth), normalized by the slate size, then averaged over
    # decisions and square-rooted (matches the report RMSE_p definition).
    sse = ((pred - true_probs_val) ** 2).sum(axis=1)
    rmse_probs = float(np.sqrt(np.mean(sse / slate_val.sum(axis=1))))

    return {
        "seed": seed,
        "val_loss": val_loss,
        "train_loss_final": float(history["train_loss"][-1]),
        "rmse_probs_vs_truth": rmse_probs,
        "train_time_sec": train_time,
    }


def main():
    """Parse CLI arguments, run the experiment over all seeds, and log results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="32,33,34,35,36,37,38,39,40,41")
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--N_t", type=int, default=50)
    parser.add_argument("--K_min", type=int, default=2)
    parser.add_argument("--K_max", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    run_id = get_run_id()
    out = results_dir("1_dh_only_baseline")
    config = dict(T=args.T, N_t=args.N_t, K_min=args.K_min, K_max=args.K_max,
                  epochs=args.epochs, batch_size=args.batch_size)
    config_hash = get_config_hash(config)

    print(f"=== 1_dh_only_baseline  run_id={run_id}  config={config_hash} ===")
    print(f"  T={args.T} N_t={args.N_t} K=[{args.K_min},{args.K_max}]  seeds={seeds}")
    print("  utility: V_j = h_DH(1_S)_j   (no macro)")

    rows = []
    for seed in seeds:
        r = run_one(seed, args.T, args.N_t, args.K_min, args.K_max,
                    args.epochs, args.batch_size)
        r.update({"run_id": run_id, "config_hash": config_hash, **config})
        rows.append(r)
        print(f"  seed={seed}  val_loss={r['val_loss']:.4f}  "
              f"RMSE_probs={r['rmse_probs_vs_truth']:.4f}  "
              f"train_loss={r['train_loss_final']:.4f}  time={r['train_time_sec']:.1f}s")
        append_log(
            out / "log.csv", row=r,
            fields=["run_id", "config_hash", "T", "N_t", "K_min", "K_max",
                    "epochs", "batch_size", "seed",
                    "val_loss", "train_loss_final",
                    "rmse_probs_vs_truth", "train_time_sec"],
        )

    val = np.array([r["val_loss"] for r in rows])
    rmse = np.array([r["rmse_probs_vs_truth"] for r in rows])
    print(f"\n=== Summary ({len(seeds)} seeds) ===")
    print(f"  val_loss        = {val.mean():.4f} +/- {val.std():.4f}")
    print(f"  RMSE (vs truth) = {rmse.mean():.4f} +/- {rmse.std():.4f}")
    print(f"  Logged to {out}/log.csv")


if __name__ == "__main__":
    main()
