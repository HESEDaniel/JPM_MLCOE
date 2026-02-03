"""Performance measurement and experiment utilities."""

import csv
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import tensorflow as tf


def setup_gpu(gpu_id=None):
    """Set up GPU with memory growth and optional device selection.

    This prevents TensorFlow from allocating all GPU memory at once,
    allowing multiple experiments to run concurrently.

    Parameters
    ----------
    gpu_id : int, optional
        GPU device ID to use. If specified, sets CUDA_VISIBLE_DEVICES.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def set_seeds(seed):
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed for numpy and TensorFlow.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


@contextmanager
def measure_time():
    """Context manager to measure elapsed time.

    Usage:
        with measure_time() as metrics:
            model.fit(dataset)
        print(metrics["elapsed_time"])

    Yields:
        dict: Dictionary that will be populated with 'elapsed_time' (seconds).
    """
    start_time = time.time()
    metrics = {}
    yield metrics
    metrics["elapsed_time"] = time.time() - start_time


def get_config_hash(config_dict):
    """Generate MD5 hash from configuration dictionary for deduplication.

    Args:
        config_dict: Dictionary of configuration parameters.

    Returns:
        str: First 12 characters of MD5 hash of the config.
    """
    return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:12]


def load_experiment_log(log_file):
    """Load existing experiments from CSV log file.

    Args:
        log_file: Path to CSV log file.

    Returns:
        dict: Dictionary mapping config_hash to experiment data.
    """
    log_file = Path(log_file)
    experiments = {}
    if log_file.exists():
        with open(log_file) as f:
            for row in csv.DictReader(f):
                if row.get("config_hash"):
                    experiments[row["config_hash"]] = row
    return experiments


def save_experiment_log(log_file, experiment, log_fields):
    """Save experiment to CSV log file.

    Args:
        log_file: Path to CSV log file.
        experiment: Dictionary of experiment data to save.
        log_fields: List of field names for CSV columns.
    """
    if not experiment:
        return

    log_file = Path(log_file)
    experiments = load_experiment_log(log_file)
    experiments[experiment["config_hash"]] = experiment

    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields, extrasaction="ignore")
        writer.writeheader()
        for exp in sorted(experiments.values(), key=lambda x: x.get("run_id", "")):
            writer.writerow(exp)


