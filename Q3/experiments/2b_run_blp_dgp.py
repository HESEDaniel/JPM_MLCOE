"""Run BLP baseline experiments on Lu25 DGP 1-4.

Usage:
    python 2b_run_blp_dgp.py --dgp dgp1 dgp2 --iv cost_iv --n_rep 50
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyblp
pyblp.options.verbose = False

from datasets.dgp import simulate_lu25_dgp
from models.pyblp import PyBLPEstimator

RESULTS_DIR = Path(__file__).parent / "results" / "blp"


def run_single(dgp_type, T, J, N, iv_type, seed):
    """Run a single BLP experiment.

    Args:
        dgp_type: "dgp1"-"dgp4"
        T, J, N: number of markets, products, consumer draws
        iv_type: "cost_iv" or "weak_iv"
        seed: random seed
    """
    X, q, true_params = simulate_lu25_dgp(T=T, J=J, N=N, dgp_type=dgp_type, seed=seed)
    cost_shock = true_params["cost_shock"] if iv_type == "cost_iv" else None

    t0 = time.time()
    model = PyBLPEstimator(iv_type=iv_type, seed=seed)
    model.fit(X, q, cost_shock=cost_shock)
    result = model.get_results(true_params)
    elapsed = time.time() - t0

    result.update({
        "dgp": dgp_type,
        "iv_type": iv_type,
        "method": "pyblp",
        "T": T,
        "J": J,
        "seed": seed,
        "elapsed_sec": round(elapsed, 2),
        "beta_p_true": float(true_params["beta_bar"][0]),
        "beta_w_true": float(true_params["beta_bar"][1]),
        "sigma_true": float(true_params["sigma"][0]),
    })
    return result


def run_experiments(dgps, ivs, T, J, N, n_rep, output_dir, seed_start=42):
    """Run all experiments and save results incrementally to CSV.

    Args:
        dgps: list of DGP types, e.g. ["dgp1", "dgp2"]
        ivs: list of IV types, e.g. ["cost_iv", "weak_iv"]
        T, J, N: number of markets, products, consumer draws
        n_rep: number of replications per (dgp, iv) combination
        output_dir: directory to save results
        seed_start: starting seed (seeds = seed_start, ..., seed_start + n_rep - 1)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_csv = output_dir / "blp_results.csv"

    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    config = {
        "dgps": dgps, "ivs": ivs,
        "T": T, "J": J, "N": N, "n_rep": n_rep,
        "seed_start": seed_start, "timestamp": timestamp,
    }
    with open(config_dir / f"blp_{timestamp}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    master_exists = master_csv.exists() and master_csv.stat().st_size > 0
    all_results = []
    total = len(dgps) * len(ivs) * n_rep
    count = 0

    for dgp in dgps:
        for iv in ivs:
            for rep in range(n_rep):
                count += 1
                seed = seed_start + rep

                try:
                    result = run_single(dgp, T, J, N, iv, seed)
                    all_results.append(result)
                except Exception as e:
                    result = {"dgp": dgp, "iv_type": iv, "method": "pyblp",
                              "seed": seed, "error": str(e)}
                    all_results.append(result)

                row_df = pd.DataFrame([result])
                row_df.to_csv(master_csv, mode='a', header=not master_exists, index=False)
                master_exists = True

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp", nargs="+", default=["dgp1", "dgp2", "dgp3", "dgp4"],
                        choices=["dgp1", "dgp2", "dgp3", "dgp4"])
    parser.add_argument("--iv", nargs="+", default=["cost_iv", "weak_iv"],
                        choices=["cost_iv", "weak_iv"])
    parser.add_argument("--T", type=int, default=25)
    parser.add_argument("--J", type=int, default=5)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--n_rep", type=int, default=1)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    run_experiments(
        dgps=args.dgp, ivs=args.iv,
        T=args.T, J=args.J, N=args.N,
        n_rep=args.n_rep,
        output_dir=args.output_dir, seed_start=args.seed_start,
    )


if __name__ == "__main__":
    main()
