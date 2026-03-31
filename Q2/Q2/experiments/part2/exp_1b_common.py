"""Part 2-1b: Shared utilities for PFPF comparison experiments."""
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.flows import LEDHFlow, PFPFLEDHFilter
from src.filters.common import DTYPE


def exp_lambda_schedule(nf, ratio=1.2):
    raw = tf.constant([ratio**i for i in range(nf)], dtype=DTYPE)
    cs = tf.cumsum(raw)
    return tf.concat([[0.0], cs / cs[-1]], axis=0)


def build_configs(N, nf, lam=None, Q_diff=None):
    """Build 4-way comparison for report.

    1. LEDH standalone (ODE flow, no importance weights)
    2. PFPF-LEDH li17 (ODE proposal, full importance weights)
    3. PFPF-LEDH dai22 detJ Q=0 (Dai22 deterministic proposal)
    4. Dai22 SDE Q>0 (standalone stochastic flow, uniform weights)

    Q_diff controls #4: if None, only 3 configs (no SDE standalone).
    """
    from src.flows import PFPFEDHFilter
    from src.flows.stochastic_flow import StochasticFlow

    kw = {'lambda_schedule': lam} if lam is not None else {}
    configs = [
        ('LEDH',               lambda: LEDHFlow(N, nf, **kw)),
        ('PFPF-LEDH (li17)',   lambda: PFPFLEDHFilter(N, nf, flow_type='li17', **kw)),
        ('PFPF-LEDH (dai22)',  lambda: PFPFLEDHFilter(N, nf, flow_type='dai22',
                                          weight_type='det_jacobian', **kw)),
    ]
    if Q_diff is not None:
        configs.append(
            ('Dai22 SDE',      lambda: StochasticFlow(N, nf, Q_diff=Q_diff, **kw)),
        )
    return configs


def _safe_name(name):
    return name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')


def run_experiment(exp_name, ssm, configs, n_traj, n_trials,
                   simulate_fn, metric_ts_fn, metric_name, base_dir,
                   seed=42, config_dict=None):
    """Run experiment with timestamped run directory and CSV logging.

    Directory structure:
        base_dir/
        +-- runs/
        |   +-- 2026-03-27_01-23-45/
        |       +-- config.json
        |       +-- data/           # per-algorithm npz [n_runs, T]
        |       |   +-- LEDH.npz
        |       |   +-- ...
        |       +-- metrics/
        |           +-- summary.txt
        +-- experiment_log.csv      # appended each run
    """
    # Dedup check
    if config_dict is not None:
        from src.utils.config_manager import find_existing_run
        runs_dir = os.path.join(base_dir, 'runs')
        existing = find_existing_run(runs_dir, config_dict)
        if existing is not None:
            print(f"Duplicate run found at {existing}, skipping.")
            return None

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(base_dir, 'runs', timestamp)
    data_dir = os.path.join(run_dir, 'data')
    metrics_dir = os.path.join(run_dir, 'metrics')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Save config
    if config_dict is not None:
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

    total = n_traj * n_trials
    print(f"\n{'=' * 70}")
    print(f"{exp_name}: {n_traj} traj x {n_trials} trials = {total} runs")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 70}")

    results = {name: {'metric_ts': [], 'ess_ts': [], 'time': [], 'ok': []}
               for name, _ in configs}

    rng_np = np.random.default_rng(seed)
    t_start = time.perf_counter()

    for traj in range(n_traj):
        xs, ys = simulate_fn(ssm, rng_np)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        for trial in range(n_trials):
            for name, make_flow in configs:
                flow = make_flow()
                rng = tf.random.Generator.from_seed(traj * 100 + trial)
                t0 = time.perf_counter()
                try:
                    res = flow.filter(ssm, ys_tf, rng=rng)
                    dt = time.perf_counter() - t0
                    metric_ts = metric_ts_fn(xs, res)
                    ess_ts = res.diagnostics['ess'].numpy()
                    results[name]['metric_ts'].append(metric_ts)
                    results[name]['ess_ts'].append(ess_ts)
                    results[name]['time'].append(dt)
                    results[name]['ok'].append(True)
                except Exception as e:
                    dt = time.perf_counter() - t0
                    T_len = ys.shape[0] if hasattr(ys, 'shape') else len(ys)
                    results[name]['metric_ts'].append(np.full(T_len, np.nan))
                    results[name]['ess_ts'].append(np.full(T_len, np.nan))
                    results[name]['time'].append(dt)
                    results[name]['ok'].append(False)

            # Save after each trial
            _save_data(results, configs, data_dir)

            run_idx = traj * n_trials + trial + 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / run_idx * (total - run_idx) / 60
            print(f"  [{run_idx}/{total}] traj={traj+1} trial={trial+1}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta:.1f}min")

    total_time = time.perf_counter() - t_start

    # Save summary to metrics/
    _save_summary(results, configs, metric_name, metrics_dir, exp_name)

    # Print summary
    _print_summary(results, configs, metric_name)

    # Append to experiment_log.csv
    _append_log(results, configs, metric_name, base_dir,
                timestamp, exp_name, config_dict, total_time)

    print(f"\nSaved to {run_dir}")
    return results


def _save_data(results, configs, data_dir):
    """Save per-algorithm npz with per-timestep data."""
    for name, _ in configs:
        r = results[name]
        np.savez(
            os.path.join(data_dir, f'{_safe_name(name)}.npz'),
            metric_ts=np.array(r['metric_ts']) if r['metric_ts'] else np.array([]),
            ess_ts=np.array(r['ess_ts']) if r['ess_ts'] else np.array([]),
            runtime=np.array(r['time']),
            ok=np.array(r['ok']))


def _save_summary(results, configs, metric_name, metrics_dir, exp_name):
    """Save summary.txt to metrics/."""
    with open(os.path.join(metrics_dir, 'summary.txt'), 'w') as f:
        f.write(f"{exp_name}\n{'=' * 60}\n\n")
        f.write(f"{'Algorithm':<24s}  {metric_name:>8s}  {'std':>8s}  "
                f"{'ESS':>8s}  {'Time':>7s}  {'OK':>5s}\n")
        f.write("-" * 65 + "\n")
        for name, _ in configs:
            r = results[name]
            ok = [i for i, v in enumerate(r['ok']) if v]
            if ok:
                means = [np.nanmean(r['metric_ts'][i]) for i in ok]
                ess_means = [np.nanmean(r['ess_ts'][i]) for i in ok]
                f.write(f"{name:<24s}  {np.mean(means):8.4f}  {np.std(means):8.4f}  "
                        f"{np.mean(ess_means):8.1f}  "
                        f"{np.mean([r['time'][i] for i in ok]):7.1f}  "
                        f"{len(ok)}/{len(r['ok'])}\n")
            else:
                f.write(f"{name:<24s}  FAIL\n")


def _print_summary(results, configs, metric_name):
    """Print summary table to console."""
    print(f"\n{'Algorithm':<24s}  {metric_name:>8s}  {'std':>8s}  "
          f"{'ESS':>8s}  {'Time':>7s}  {'OK':>5s}")
    print("-" * 65)
    for name, _ in configs:
        r = results[name]
        ok = [i for i, v in enumerate(r['ok']) if v]
        if ok:
            means = [np.nanmean(r['metric_ts'][i]) for i in ok]
            ess_means = [np.nanmean(r['ess_ts'][i]) for i in ok]
            print(f"{name:<24s}  {np.mean(means):8.4f}  {np.std(means):8.4f}  "
                  f"{np.mean(ess_means):8.1f}  "
                  f"{np.mean([r['time'][i] for i in ok]):7.1f}  "
                  f"{len(ok)}/{len(r['ok'])}")
        else:
            print(f"{name:<24s}  {'FAIL':>8s}{'':>28s}  0/{len(r['ok'])}")


def _append_log(results, configs, metric_name, base_dir,
                timestamp, exp_name, config_dict, total_time):
    """Append one row per algorithm to experiment_log.csv."""
    log_path = os.path.join(base_dir, 'experiment_log.csv')
    write_header = not os.path.exists(log_path)

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'timestamp', 'experiment', 'algorithm',
                f'mean_{metric_name.lower()}', f'std_{metric_name.lower()}',
                'mean_ess', 'mean_runtime', 'ok', 'total',
                'total_time_sec', 'config'])

        config_str = json.dumps(config_dict) if config_dict else ''
        for name, _ in configs:
            r = results[name]
            ok = [i for i, v in enumerate(r['ok']) if v]
            if ok:
                means = [np.nanmean(r['metric_ts'][i]) for i in ok]
                ess_means = [np.nanmean(r['ess_ts'][i]) for i in ok]
                writer.writerow([
                    timestamp, exp_name, name,
                    f'{np.mean(means):.4f}', f'{np.std(means):.4f}',
                    f'{np.mean(ess_means):.1f}',
                    f'{np.mean([r["time"][i] for i in ok]):.1f}',
                    len(ok), len(r['ok']),
                    f'{total_time:.1f}', config_str])
            else:
                writer.writerow([
                    timestamp, exp_name, name,
                    'NaN', 'NaN', 'NaN', 'NaN',
                    0, len(r['ok']),
                    f'{total_time:.1f}', config_str])
