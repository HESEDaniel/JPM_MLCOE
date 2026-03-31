"""Experiment configuration management: save, find, and dedup."""
import json
import os
from datetime import datetime


def save_config(results_dir, config):
    """Save config dict as config.json with automatic timestamp.

    Returns the path to the saved config file.
    """
    config = dict(config)
    config.setdefault('timestamp', datetime.now().isoformat())
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    return path


def find_existing_run(search_dir, config, ignore_keys=('timestamp',)):
    """Search subdirectories of search_dir for a matching config.json.

    Returns the path to the matching directory, or None.
    """
    if not os.path.isdir(search_dir):
        return None

    target = {k: v for k, v in config.items() if k not in ignore_keys}

    for entry in os.scandir(search_dir):
        if not entry.is_dir():
            continue
        cfg_path = os.path.join(entry.path, 'config.json')
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path) as f:
                existing = json.load(f)
            existing_filtered = {k: v for k, v in existing.items()
                                 if k not in ignore_keys}
            if existing_filtered == target:
                return entry.path
        except (json.JSONDecodeError, IOError):
            continue
    return None


def check_and_save(results_dir, config, search_dir=None):
    """Find existing run or save new config.

    Returns (path, is_duplicate).
    """
    if search_dir is None:
        search_dir = os.path.dirname(results_dir)

    existing = find_existing_run(search_dir, config)
    if existing is not None:
        return existing, True

    save_config(results_dir, config)
    return results_dir, False
