"""Run FeatureBasedDeepHalo experiments.

Usage:
    python run_featurebased_deephalo.py --dataset heating
    python run_featurebased_deephalo.py --dataset synthetic --n_items 8 --n_features 5
    python run_featurebased_deephalo.py --dataset heating --embedding_dim 32 --n_layers 3 --n_heads 4
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
from choice_learn.data import ChoiceDataset
from choice_learn.datasets.base import load_heating
from choice_learn.models import FeatureBasedDeepHalo

from utils.utils import (
    setup_gpu, get_config_hash, load_experiment_log, save_experiment_log
)

setup_gpu()

LOG_FIELDS = [
    "run_id", "config_hash", "dataset", "n_items", "n_features", "n_choices",
    "embedding_dim", "n_layers", "n_heads", "seed",
    "epochs", "batch_size", "lr", "n_params",
    "final_loss", "final_accuracy", "timestamp",
]


def load_heating_dataset():
    """Load heating dataset for FeatureBasedDeepHalo."""
    heating_df = load_heating(as_frame=True)
    items = ["hp", "gc", "gr", "ec", "er"]

    choices = np.array([items.index(val) for val in heating_df["depvar"].to_numpy().ravel()])
    items_features = np.stack(
        [heating_df[["ic." + item, "oc." + item]].to_numpy() for item in items],
        axis=1,
    ).astype("float32")

    return ChoiceDataset(items_features_by_choice=items_features, choices=choices)


def create_synthetic_dataset(n_choices, n_items, n_features, seed):
    """Create synthetic dataset with random features."""
    np.random.seed(seed)

    items_features = np.random.randn(n_choices, n_items, n_features).astype(np.float32)
    available = np.ones((n_choices, n_items), dtype=np.float32)
    choices = np.random.randint(0, n_items, n_choices)

    return ChoiceDataset(
        items_features_by_choice=items_features,
        available_items_by_choice=available,
        choices=choices,
    )


def compute_accuracy(model, dataset):
    """Compute prediction accuracy."""
    probas = model.predict_probas(dataset).numpy()
    predictions = np.argmax(probas, axis=1)
    return np.mean(predictions == dataset.choices)


def run_experiment(
    dataset_name, n_items, n_features, n_choices,
    embedding_dim, n_layers, n_heads,
    epochs, batch_size, lr, seed, output_dir, force=False,
):
    """Run single experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment_log.csv"

    # Check for existing run
    config = {
        "dataset": dataset_name, "embedding_dim": embedding_dim, "n_layers": n_layers,
        "n_heads": n_heads, "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
    }
    config_hash = get_config_hash(config)
    experiments = load_experiment_log(log_file)

    if config_hash in experiments and not force:
        print(f"Skipping: {config_hash} (use --force)")
        return None

    run_id = datetime.now().strftime("%y%m%d_%H%M%S")

    # Load dataset
    if dataset_name == "heating":
        dataset = load_heating_dataset()
        n_items = dataset.get_n_items()
        items_features = dataset.items_features_by_choice
        if isinstance(items_features, tuple):
            n_features = sum(f.shape[-1] for f in items_features)
        else:
            n_features = items_features.shape[-1]
        n_choices = dataset.get_n_choices()
    else:
        dataset = create_synthetic_dataset(n_choices, n_items, n_features, seed)

    # Create model
    tf.random.set_seed(seed)
    model = FeatureBasedDeepHalo(
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        optimizer="adam",
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Train
    history = model.fit(dataset, verbose=0)

    n_params = sum(np.prod(w.shape) for w in model.trainable_weights)
    final_loss = history["train_loss"][-1]
    final_accuracy = compute_accuracy(model, dataset)

    # Save results
    json_dir = output_dir / "json"
    weights_dir = output_dir / "weights"
    json_dir.mkdir(exist_ok=True)
    weights_dir.mkdir(exist_ok=True)

    # Save weights
    weights_file = weights_dir / f"{dataset_name}_d{embedding_dim}_L{n_layers}_H{n_heads}_{run_id}.weights.npz"
    np.savez(weights_file, **{f"w{i}": w.numpy() for i, w in enumerate(model.trainable_weights)})

    # Save detailed results
    result = {
        "run_id": run_id,
        "config_hash": config_hash,
        "dataset": dataset_name,
        "n_items": int(n_items),
        "n_features": int(n_features),
        "n_choices": int(n_choices),
        "embedding_dim": embedding_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_params": int(n_params),
        "final_loss": float(final_loss),
        "final_accuracy": float(final_accuracy),
        "loss_history": [float(l) for l in history["train_loss"]],
        "timestamp": datetime.now().isoformat(),
    }

    json_file = json_dir / f"{dataset_name}_d{embedding_dim}_L{n_layers}_H{n_heads}_{run_id}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)

    save_experiment_log(log_file, result, LOG_FIELDS)
    print(f"Saved: {json_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run FeatureBasedDeepHalo experiments")
    parser.add_argument("--dataset", "-D", type=str, choices=["heating", "synthetic"], default="heating",
                        help="Dataset to use (default: heating)")
    parser.add_argument("--n_items", type=int, default=8, help="Number of items for synthetic dataset")
    parser.add_argument("--n_features", type=int, default=5, help="Number of features for synthetic dataset")
    parser.add_argument("--n_choices", type=int, default=1000, help="Number of choices for synthetic dataset")
    parser.add_argument("--embedding_dim", "-d", type=int, default=16, help="Embedding dimension (default: 16)")
    parser.add_argument("--n_layers", "-L", type=int, default=2, help="Number of layers (default: 2)")
    parser.add_argument("--n_heads", "-H", type=int, default=4, help="Number of heads (default: 4)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default="./experiments/results/featurebased",
                        help="Output directory (default: ./experiments/results/featurebased)")
    parser.add_argument("--force", action="store_true", help="Re-run even if config exists")
    args = parser.parse_args()

    run_experiment(
        dataset_name=args.dataset,
        n_items=args.n_items,
        n_features=args.n_features,
        n_choices=args.n_choices,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
