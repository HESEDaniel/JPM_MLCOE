"""Credit card offer simulation: context effects demonstration.

Usage:
    python experiments/credit_card_simulation.py --seed 0
    python experiments/credit_card_simulation.py --seed 1 --force
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
from choice_learn.data import ChoiceDataset
from choice_learn.models import FeatureBasedDeepHalo, SimpleMNL
from choice_learn.models.halo_mnl import HaloMNL

from utils.utils import set_seeds, setup_gpu, measure_time, get_config_hash


# Credit card offers
# [discount_pct, min_spend, annual_fee, brand_score]
OFFER_FEATURES = np.array([
    [0.10, 0.5, 0.3, 0.9],   # 0: Travel Premium
    [0.06, 0.3, 0.1, 0.6],   # 1: Travel Standard
    [0.08, 0.6, 0.4, 0.7],   # 2: Travel Decoy
    [0.09, 0.4, 0.2, 0.85],  # 3: Shopping Elite
    [0.08, 0.3, 0.1, 0.7],   # 4: Shopping Plus
    [0.07, 0.4, 0.2, 0.8],   # 5: Dining Gold
    [0.05, 0.2, 0.0, 0.6],   # 6: Dining Basic
    [0.04, 0.1, 0.0, 0.5],   # 7: Gas Rewards
], dtype=np.float32)

BETA_WEIGHTS = np.array([8.0, -2.0, -3.0, 5.0], dtype=np.float32)


def generate_credit_card_choices(n_choices, seed=42):
    """Generate synthetic credit card choice data with context effects.

    Parameters
    ----------
    n_choices : int
        Number of choice observations to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (ChoiceDataset, true_probs array)
    """
    rng = np.random.RandomState(seed)
    n_items, n_features = OFFER_FEATURES.shape

    items_features = np.zeros((n_choices, n_items, n_features), dtype=np.float32)
    available = np.zeros((n_choices, n_items), dtype=np.float32)
    choices = np.zeros(n_choices, dtype=np.int32)
    true_probs = np.zeros((n_choices, n_items), dtype=np.float32)

    for i in range(n_choices):
        n_avail = rng.randint(4, 7)
        avail_idx = np.sort(rng.choice(n_items, size=n_avail, replace=False))
        available[i, avail_idx] = 1.0
        items_features[i] = OFFER_FEATURES

        utilities = OFFER_FEATURES[avail_idx] @ BETA_WEIGHTS
        # Context effect: Travel Premium (0) boosted, Travel Decoy (2) penalized when both present
        if 0 in avail_idx and 2 in avail_idx:
            utilities[np.where(avail_idx == 0)[0][0]] += 1.5
            utilities[np.where(avail_idx == 2)[0][0]] -= 1.5

        utilities -= utilities.max()
        probs = np.exp(utilities) / np.exp(utilities).sum()
        true_probs[i, avail_idx] = probs
        choices[i] = avail_idx[rng.choice(len(avail_idx), p=probs)]

    dataset = ChoiceDataset(
        items_features_by_choice=items_features,
        available_items_by_choice=available,
        choices=choices,
    )
    return dataset, true_probs


def compute_rmse(pred_probs, true_probs, available):
    """Compute RMSE between predicted and true probabilities.

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted probabilities.
    true_probs : np.ndarray
        True probabilities.
    available : np.ndarray
        Availability mask.

    Returns
    -------
    float
        Root mean squared error.
    """
    mask = available.astype(bool)
    diff = (pred_probs - true_probs)[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def get_model_probs(model, dataset):
    """Get predicted probabilities from a choice model.

    Parameters
    ----------
    model : ChoiceModel
        Trained choice model.
    dataset : ChoiceDataset
        Dataset to predict on.

    Returns
    -------
    np.ndarray
        Predicted probabilities.
    """
    for batch in dataset.iter_batch(batch_size=-1):
        utilities = model.compute_batch_utility(*batch)
        utilities = tf.where(batch[2] > 0, utilities, -1e9)
        return tf.nn.softmax(utilities, axis=-1).numpy()


def warmup():
    """Run each model once to trigger TensorFlow compilation."""
    dataset, _ = generate_credit_card_choices(100, seed=0)
    for Model, kwargs in [
        (SimpleMNL, {"intercept": "item"}),
        (HaloMNL, {"intercept": "item"}),
        (FeatureBasedDeepHalo, {"embedding_dim": 8, "n_layers": 2, "n_heads": 2}),
    ]:
        model = Model(**kwargs, optimizer="adam", lr=1e-3, epochs=1, batch_size=32)
        model.fit(dataset, verbose=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID")
    parser.add_argument("--n_choices", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./experiments/results/credit_card")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    setup_gpu(args.gpu)

    config = {
        "n_choices": args.n_choices,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embedding_dim": args.embedding_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
    }
    config_hash = get_config_hash(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"seed{args.seed}_{config_hash}.json"

    # Check if exists
    if out_file.exists() and not args.force:
        print(f"Results exist: {out_file}. Use --force to re-run.")
        return

    # Run
    set_seeds(args.seed)
    warmup()

    dataset, true_probs = generate_credit_card_choices(args.n_choices, seed=args.seed)
    avail = dataset.available_items_by_choice
    n_items = dataset.get_n_items()
    n_choices = dataset.get_n_choices()

    results = {"config": config, "seed": args.seed}

    # Random
    random_probs = np.zeros((n_choices, n_items), dtype=np.float32)
    for i in range(n_choices):
        mask = avail[i].astype(bool)
        random_probs[i, mask] = 1.0 / mask.sum()
    results["Random"] = {"rmse": compute_rmse(random_probs, true_probs, avail)}

    # SimpleMNL
    mnl = SimpleMNL(intercept="item", optimizer="adam", lr=args.lr,
                    epochs=args.epochs, batch_size=args.batch_size)
    with measure_time() as t:
        mnl.fit(dataset, verbose=0)
    results["SimpleMNL"] = {
        "rmse": compute_rmse(get_model_probs(mnl, dataset), true_probs, avail),
        "time": t["elapsed_time"],
    }

    # HaloMNL
    halo = HaloMNL(intercept="item", optimizer="adam", lr=args.lr,
                   epochs=args.epochs, batch_size=args.batch_size)
    with measure_time() as t:
        halo.fit(dataset, verbose=0)
    results["HaloMNL"] = {
        "rmse": compute_rmse(get_model_probs(halo, dataset), true_probs, avail),
        "time": t["elapsed_time"],
    }

    # DeepHalo
    deep = FeatureBasedDeepHalo(
        embedding_dim=args.embedding_dim, n_layers=args.n_layers, n_heads=args.n_heads,
        optimizer="adam", lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    with measure_time() as t:
        deep.fit(dataset, verbose=0)
    results["DeepHalo"] = {
        "rmse": compute_rmse(get_model_probs(deep, dataset), true_probs, avail),
        "time": t["elapsed_time"],
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
