"""Light helper: config-hash based result logging.

Usage:
    from src._results import get_config_hash, append_log

    config_hash = get_config_hash(config_dict)
    append_log(Path("results/<experiment-name>/log.csv"),
               row={"config_hash": config_hash, "model": "...", "val_loss": 0.96, ...},
               fields=["config_hash", "model", "val_loss", ...])
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


def set_all_seeds(seed: int) -> None:
    """Seed EVERY global RNG source the pipeline touches: python ``random``,
    global ``numpy``, and ``tensorflow``.

    The DGP is reproducible on its own because it uses a *local*
    ``np.random.default_rng(seed)``. But the choice-learn training reads the
    GLOBAL numpy / python RNG (e.g. for batch shuffling), so seeding only
    ``tf.random.set_seed`` leaves the model fit non-reproducible across
    processes. ``top_k`` (an argmax) then amplifies the tiny weight differences
    into visibly different recall. Call this instead of ``tf.random.set_seed``.
    """
    import random as _random

    import numpy as _np
    import tensorflow as _tf

    _random.seed(seed)
    _np.random.seed(seed)
    _tf.random.set_seed(seed)


def get_config_hash(config: dict, length: int = 10) -> str:
    """Stable hash of a config dict (ordered JSON serialization)."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:length]


def get_run_id() -> str:
    """Timestamp run-id, microsecond resolution -> naturally sortable.

    Format: 'YYMMDD_HHMMSS_ffffff'  (e.g., '260521_153045_837291')
    """
    from datetime import datetime
    return datetime.now().strftime("%y%m%d_%H%M%S_%f")


def append_log(log_path: Path, row: dict, fields: Iterable[str]) -> None:
    """Append a row to a CSV log. Writes header on first row."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def results_dir(experiment_name: str) -> Path:
    """Standard results dir for an experiment: results/<experiment_name>/."""
    root = Path(__file__).resolve().parent.parent / "results" / experiment_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    return root
