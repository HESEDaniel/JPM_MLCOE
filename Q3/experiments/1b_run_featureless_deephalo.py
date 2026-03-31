"""Run FeaturelessDeepHalo experiments from Zhang et al. (2025).

Usage:
    python run_featureless_deephalo.py --layer 5 --param 200k
    python run_featureless_deephalo.py --all
    python run_featureless_deephalo.py --layer 3 --param 200k --force
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
from choice_learn.data import ChoiceDataset
from choice_learn.models import FeaturelessDeepHalo

from utils.utils import (
    setup_gpu, get_config_hash, load_experiment_log, save_experiment_log
)

setup_gpu()

# Constants: FeaturelessDeepHalo configurations from Zhang et al. (2025)
# param_budget -> {layer: (width, paper_rmse)}
CONFIGS = {
    "200k": {3: (306, 0.0434), 4: (251, 0.0243), 5: (218, 0.0202), 6: (195, 0.0188), 7: (179, 0.0183)},
    "500k": {3: (489, 0.0419), 4: (401, 0.0156), 5: (348, 0.0140), 6: (312, 0.0131), 7: (285, 0.0130)},
}

# Paper reference RMSE values
PAPER_RMSE = {
    "200k": {3: 0.0434, 4: 0.0243, 5: 0.0202, 6: 0.0188, 7: 0.0183},
    "500k": {3: 0.0419, 4: 0.0156, 5: 0.0140, 6: 0.0131, 7: 0.0130},
}

LOG_FIELDS = [
    "run_id", "config_hash", "layer", "width", "param", "seed", "init",
    "epochs", "batch_size", "lr", "n_params", "rmse_true", "rmse_emp", "paper_rmse", "timestamp",
]


def compute_empirical_freq(available, choices, n_items):
    """Compute empirical frequencies by averaging one-hot choices per choice set."""
    n_samples = len(choices)

    # One-hot encode choices
    one_hot = np.zeros((n_samples, n_items), dtype=np.float32)
    one_hot[np.arange(n_samples), choices] = 1.0

    # Group by availability pattern and average
    pattern_indices = defaultdict(list)
    for i, row in enumerate(available):
        pattern_indices[tuple(row)].append(i)

    emp_freq = np.zeros((n_samples, n_items), dtype=np.float32)
    for indices in pattern_indices.values():
        emp_freq[indices] = np.mean(one_hot[indices], axis=0)

    return emp_freq


def generate_synthetic_data(n_items=20, choice_set_size=15, samples_per_set=80, seed=42):
    """Generate synthetic data with all C(n_items, choice_set_size) choice sets."""
    np.random.seed(seed)

    choice_sets = list(itertools.combinations(range(n_items), choice_set_size))
    n_samples = len(choice_sets) * samples_per_set

    available = np.zeros((n_samples, n_items), dtype=np.float32)
    choices = np.zeros(n_samples, dtype=np.int32)
    true_probs = np.zeros((n_samples, n_items), dtype=np.float32)

    idx = 0
    for choice_set in choice_sets:
        choice_set = list(choice_set)
        probs = np.random.dirichlet(np.ones(choice_set_size))
        for _ in range(samples_per_set):
            available[idx, choice_set] = 1.0
            choices[idx] = choice_set[np.random.choice(choice_set_size, p=probs)]
            true_probs[idx, choice_set] = probs
            idx += 1

    emp_freq = compute_empirical_freq(available, choices, n_items)
    dataset = ChoiceDataset(available_items_by_choice=available, choices=choices)

    return dataset, true_probs, emp_freq


def train_and_evaluate(model, dataset, true_probs, emp_freq, epochs, batch_size):
    """Train model and return RMSE history."""
    rmse_true_history, rmse_emp_history = [], []
    n_samples = len(dataset)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        losses = []

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            loss = model.train_step(
                None, None,
                dataset.available_items_by_choice[batch_idx],
                dataset.choices[batch_idx],
            )
            losses.append(float(loss))

        pred = model.predict_probas(dataset).numpy()
        rmse_true = np.sqrt(np.mean((pred - true_probs) ** 2))
        rmse_emp = np.sqrt(np.mean((pred - emp_freq) ** 2))
        rmse_true_history.append(rmse_true)
        rmse_emp_history.append(rmse_emp)


    return rmse_true_history, rmse_emp_history


def run_experiment(layer, param, epochs, batch_size, lr, seed, init, output_dir, force=False):
    """Run single experiment."""
    width, paper_rmse = CONFIGS[param][layer]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment_log.csv"

    # Check for existing run
    config = {
        "layer": layer, "param": param, "seed": seed,
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "init": init,
    }
    config_hash = get_config_hash(config)
    experiments = load_experiment_log(log_file)

    if config_hash in experiments and not force:
        print(f"Skipping: {config_hash} (use --force)")
        return None

    run_id = datetime.now().strftime("%y%m%d_%H%M%S")

    # Generate data
    dataset, true_probs, emp_freq = generate_synthetic_data(seed=seed)

    # Create model
    tf.random.set_seed(seed)
    model = FeaturelessDeepHalo(
        n_items=20, width=width, n_layers=layer,
        init=init, optimizer="adam", lr=lr,
    )
    model.instantiate()
    n_params = sum(np.prod(w.shape) for w in model.trainable_weights)

    # Train
    rmse_true_history, rmse_emp_history = train_and_evaluate(
        model, dataset, true_probs, emp_freq, epochs, batch_size,
    )

    # Save results
    json_dir = output_dir / "json"
    weights_dir = output_dir / "weights"
    json_dir.mkdir(exist_ok=True)
    weights_dir.mkdir(exist_ok=True)

    # Save weights
    weights_file = weights_dir / f"layer{layer}_param{param}_init{init}_{run_id}.weights.npz"
    np.savez(weights_file, **{f"w{i}": w.numpy() for i, w in enumerate(model.trainable_weights)})

    # Save detailed results
    result = {
        "run_id": run_id,
        "config_hash": config_hash,
        "layer": layer,
        "width": width,
        "param": param,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "init": init,
        "n_params": int(n_params),
        "rmse_true": float(rmse_true_history[-1]),
        "rmse_emp": float(rmse_emp_history[-1]),
        "paper_rmse": paper_rmse,
        "rmse_true_history": [float(r) for r in rmse_true_history],
        "rmse_emp_history": [float(r) for r in rmse_emp_history],
        "timestamp": datetime.now().isoformat(),
    }

    json_file = json_dir / f"layer{layer}_param{param}_init{init}_{run_id}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)

    save_experiment_log(log_file, result, LOG_FIELDS)
    print(f"Saved: {json_file}")

    return result




def main():
    parser = argparse.ArgumentParser(description="Run FeaturelessDeepHalo experiments")
    parser.add_argument("--layer", "-L", type=int, choices=[3, 4, 5, 6, 7], default=3)
    parser.add_argument("--param", "-P", type=str, choices=["200k", "500k"], default="200k")
    parser.add_argument("--init", "-I", type=str, choices=["normal", "glorot", "he"], default="he",
                        help="Weight initialization (default: he)")
    parser.add_argument("--all", action="store_true", help="Run all configurations")
    parser.add_argument("--force", action="store_true", help="Re-run even if config exists")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    if args.all:
        for param in ["200k", "500k"]:
            for layer in [3, 4, 5, 6, 7]:
                run_experiment(
                    layer, param, args.epochs, args.batch_size,
                    args.lr, args.seed, args.init, args.output_dir, args.force,
                )

    elif args.layer and args.param:
        run_experiment(
            args.layer, args.param, args.epochs, args.batch_size,
            args.lr, args.seed, args.init, args.output_dir, args.force,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
