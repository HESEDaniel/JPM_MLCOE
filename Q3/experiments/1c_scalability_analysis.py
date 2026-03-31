"""Scalability analysis: benchmark DeepHalo's computational characteristics.

Usage:
    python experiments/scalability_analysis.py --seed 0
    python experiments/scalability_analysis.py --seed 1 --force
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

from utils.utils import set_seeds, setup_gpu, get_config_hash, measure_time


ITEMS_LIST = [5, 10, 20, 50, 100]
FEATURES_LIST = [4, 16, 64]
CHOICES_LIST = [500, 1000, 2000, 5000, 10000]

DEFAULT_N_ITEMS = 20
DEFAULT_N_FEATURES = 16
DEFAULT_N_CHOICES = 1000


def create_synthetic_dataset(n_items, n_features, n_choices, seed=42):
    """Create synthetic dataset with random features and availability.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_features : int
        Number of features per item.
    n_choices : int
        Number of choice observations.
    seed : int, optional
        Random seed.

    Returns
    -------
    ChoiceDataset
        Synthetic dataset.
    """
    rng = np.random.RandomState(seed)
    items_features = rng.randn(n_choices, n_items, n_features).astype(np.float32)

    min_avail = max(2, int(n_items * 0.6))
    available = np.zeros((n_choices, n_items), dtype=np.float32)
    for i in range(n_choices):
        n_avail = rng.randint(min_avail, n_items + 1)
        avail_idx = rng.choice(n_items, size=n_avail, replace=False)
        available[i, avail_idx] = 1.0

    choices = np.zeros(n_choices, dtype=np.int32)
    for i in range(n_choices):
        avail_idx = np.where(available[i] > 0)[0]
        choices[i] = rng.choice(avail_idx)

    return ChoiceDataset(
        items_features_by_choice=items_features,
        available_items_by_choice=available,
        choices=choices,
    )


def measure_single_run(model, dataset, n_epochs):
    """Measure training time for a single model run.

    Parameters
    ----------
    model : ChoiceModel
        Model to train.
    dataset : ChoiceDataset
        Training dataset.
    n_epochs : int
        Number of training epochs.

    Returns
    -------
    dict
        Dictionary with time per epoch.
    """
    with measure_time() as t:
        model.fit(dataset, verbose=0)
    return {"time": t["elapsed_time"] / max(n_epochs, 1)}


def create_model(model_name, args):
    """Create a choice model instance by name.

    Parameters
    ----------
    model_name : str
        One of "SimpleMNL", "HaloMNL", "DeepHalo".
    args : argparse.Namespace
        Command line arguments with training parameters.

    Returns
    -------
    ChoiceModel
        Instantiated model.
    """
    if model_name == "SimpleMNL":
        return SimpleMNL(intercept="item", optimizer="adam", lr=args.lr,
                         epochs=args.n_epochs, batch_size=args.batch_size)
    elif model_name == "HaloMNL":
        return HaloMNL(intercept="item", optimizer="adam", lr=args.lr,
                       epochs=args.n_epochs, batch_size=args.batch_size)
    elif model_name == "DeepHalo":
        return FeatureBasedDeepHalo(embedding_dim=16, n_layers=3, n_heads=4,
                                     optimizer="adam", lr=args.lr,
                                     epochs=args.n_epochs, batch_size=args.batch_size)


def warmup(args):
    """Run each model once to trigger TensorFlow compilation."""
    dataset = create_synthetic_dataset(5, 4, 100, seed=0)
    for model_name in ["SimpleMNL", "HaloMNL", "DeepHalo"]:
        model = create_model(model_name, args)
        model.fit(dataset, verbose=0)


def run_benchmark(n_items, n_features, n_choices, args):
    """Run benchmark for all models on given parameters.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_features : int
        Number of features.
    n_choices : int
        Number of choices.
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    dict
        Results for each model.
    """
    models = ["SimpleMNL", "HaloMNL", "DeepHalo"]
    set_seeds(args.seed)
    dataset = create_synthetic_dataset(n_items, n_features, n_choices, args.seed)

    results = {}
    for model_name in models:
        model = create_model(model_name, args)
        metrics = measure_single_run(model, dataset, args.n_epochs)
        results[model_name] = metrics
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./experiments/results/scalability")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    setup_gpu(args.gpu)

    config = {
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "items_list": ITEMS_LIST,
        "features_list": FEATURES_LIST,
        "choices_list": CHOICES_LIST,
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
    warmup(args)
    results = {"config": config, "seed": args.seed}

    # Items scaling
    results["items_scaling"] = []
    for n_items in ITEMS_LIST:
        r = run_benchmark(n_items, DEFAULT_N_FEATURES, DEFAULT_N_CHOICES, args)
        results["items_scaling"].append({"n_items": n_items, **r})

    # Features scaling
    results["features_scaling"] = []
    for n_features in FEATURES_LIST:
        r = run_benchmark(DEFAULT_N_ITEMS, n_features, DEFAULT_N_CHOICES, args)
        results["features_scaling"].append({"n_features": n_features, **r})

    # Choices scaling
    results["choices_scaling"] = []
    for n_choices in CHOICES_LIST:
        r = run_benchmark(DEFAULT_N_ITEMS, DEFAULT_N_FEATURES, n_choices, args)
        results["choices_scaling"].append({"n_choices": n_choices, **r})

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
