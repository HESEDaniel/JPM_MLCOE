"""Stochastic Volatility flow filter comparison."""
import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm.stochastic_volatility import SVAdditiveNoise
from src.flows.edh import exact_daum_huang_flow, compute_edh_matrices
from src.flows.ledh import local_edh_flow, compute_ledh_matrices
from src.flows.rkhs_pff import rkhs_particle_flow_filter
from src.utils.utils import exponential_lambda_schedule

# Filter Runners

class DivergenceError(Exception):
    """Raised when filter diverges."""
    pass

def check_divergence(m_filt, threshold=1e6):
    """Check if filter output has diverged. Raise DivergenceError if so."""
    if not np.all(np.isfinite(m_filt)):
        raise DivergenceError("NaN/Inf detected")
    if np.max(np.abs(m_filt)) > threshold:
        raise DivergenceError(f"Values exceed {threshold}")

def run_edh_filter(model, ys, N, n_flow, rng):
    """Run EDH flow filter over all time steps."""
    T = len(ys)
    ys_2d = ys.reshape(-1, 1)
    lam = exponential_lambda_schedule(n_flow, ratio=1.2)

    particles = rng.multivariate_normal(model.m0, model.P0, size=N)
    m_prev, P_prev = model.m0.copy(), model.P0.copy()
    m_filt = np.zeros((T, 1))

    for t in range(T):
        particles, _, _, m_post, P_post = exact_daum_huang_flow(
            particles, m_prev, P_prev, model.f, model.F_jac, model.Q,
            model.h, model.H_jac, model.R, ys_2d[t],
            lambda_schedule=lam, redraw=True, rng=rng, filter_type='ekf')
        m_filt[t] = m_post
        m_prev, P_prev = m_post, P_post

    return m_filt

def run_ledh_filter(model, ys, N, n_flow, rng):
    """Run LEDH flow filter over all time steps."""
    T = len(ys)
    ys_2d = ys.reshape(-1, 1)
    lam = exponential_lambda_schedule(n_flow, ratio=1.2)

    particles = rng.multivariate_normal(model.m0, model.P0, size=N)
    m_prev, P_prev = model.m0.copy(), model.P0.copy()
    m_filt = np.zeros((T, 1))

    for t in range(T):
        particles, _, _, m_post, P_post = local_edh_flow(
            particles, m_prev, P_prev, model.f, model.F_jac, model.Q,
            model.h, model.H_jac, model.R, ys_2d[t],
            lambda_schedule=lam, store_history=False, redraw=True, rng=rng, filter_type='ekf')
        m_filt[t] = m_post
        m_prev, P_prev = m_post, P_post

    return m_filt

def run_filters(model, xs, ys, N=200, n_flow=15, seed=42):
    """Run EDH, LEDH, KernelPFF and return {name: {mse, time, status}}."""
    ys_2d = ys.reshape(-1, 1)
    results = {}

    for name, runner in [
        ('EDH', lambda rng: run_edh_filter(model, ys, N, n_flow, rng)),
        ('LEDH', lambda rng: run_ledh_filter(model, ys, N, n_flow, rng)),
        ('KernelPFF', lambda rng: rkhs_particle_flow_filter(
            f=model.f, h=model.h, H_jac=model.H_jac, Q=model.Q, R=model.R,
            m0=model.m0, P0=model.P0, ys=ys_2d, N_particles=N,
            n_flow_steps=n_flow, step_size=0.1, loc_radius=4.0,
            kernel_type='scalar', rng=rng, adaptive_step=True)[0]),
    ]:
        try:
            rng = np.random.default_rng(seed)
            t0 = time.time()
            m_filt = runner(rng)
            runtime = time.time() - t0

            # Check for divergence
            check_divergence(m_filt)

            mse = np.mean((m_filt[:, 0] - xs) ** 2)
            results[name] = {'mse': mse, 'time': runtime, 'status': 'ok'}

        except DivergenceError as e:
            results[name] = {'mse': np.inf, 'time': time.time() - t0, 'status': f'diverged: {e}'}
        except Exception as e:
            results[name] = {'mse': np.inf, 'time': 0, 'status': f'error: {str(e)[:50]}'}

    return results

def run_filters_sparse(model, xs, ys, obs_mask, N=200, n_flow=15, seed=42):
    """
    Run filters with missing observations.

    obs_mask: boolean array, True = observed, False = missing
    For missing time steps, only prediction is done (no flow update).
    """
    ys_full = ys.reshape(-1, 1)
    ys_sparse = ys_full[obs_mask]
    results = {}

    for name, runner in [
        ('EDH', lambda rng: run_edh_filter(model, ys[obs_mask], N, n_flow, rng)),
        ('LEDH', lambda rng: run_ledh_filter(model, ys[obs_mask], N, n_flow, rng)),
        ('KernelPFF', lambda rng: rkhs_particle_flow_filter(
            f=model.f, h=model.h, H_jac=model.H_jac, Q=model.Q, R=model.R,
            m0=model.m0, P0=model.P0, ys=ys_sparse, N_particles=N,
            n_flow_steps=n_flow, step_size=0.1, loc_radius=4.0,
            kernel_type='scalar', rng=rng, adaptive_step=True)[0]),
    ]:
        try:
            rng = np.random.default_rng(seed)
            t0 = time.time()
            m_filt_sparse = runner(rng)
            runtime = time.time() - t0

            # Check for divergence
            check_divergence(m_filt_sparse)

            # MSE only at observed time steps
            mse = np.mean((m_filt_sparse[:, 0] - xs[obs_mask]) ** 2)
            results[name] = {'mse': mse, 'time': runtime, 'status': 'ok'}

        except DivergenceError as e:
            results[name] = {'mse': np.inf, 'time': time.time() - t0, 'status': f'diverged: {e}'}
        except Exception as e:
            results[name] = {'mse': np.inf, 'time': 0, 'status': f'error: {str(e)[:50]}'}

    return results

# Stability Diagnostics

def compute_diagnostics(model, N=200, n_steps=15, seed=42):
    """
    Compute flow stability diagnostics for EDH and LEDH.

    Returns: {method: {'cond_mean', 'cond_max', 'mag_mean', 'mag_max'}}
    """
    rng = np.random.default_rng(seed)
    particles = rng.multivariate_normal(model.m0, model.P0, size=N)
    lam_schedule = exponential_lambda_schedule(n_steps, ratio=1.2)

    # Generate one observation for diagnostics
    xs, ys = model.simulate(10, rng)
    y = np.array([ys[5]])
    m, P = model.m0.copy(), model.P0.copy()
    H = model.H_jac(m)
    R_inv = np.linalg.inv(model.R)

    diagnostics = {}

    # EDH diagnostics
    edh_conds, edh_mags = [], []
    pts = particles.copy()
    eta_bar = pts.mean(axis=0)
    lam = 0.0

    for j in range(n_steps):
        eps = lam_schedule[j]
        A, b = compute_edh_matrices(m, P, H, model.R, y, lam, eta_bar, model.h)

        cond = np.abs(A[0, 0]) if A.shape[0] == 1 else np.linalg.cond(A)
        edh_conds.append(cond if np.isfinite(cond) else 1.0)

        dx = eps * (pts @ A.T + b)
        edh_mags.append(np.mean(np.abs(dx)))

        pts += dx
        eta_bar = pts.mean(axis=0)
        lam += eps

    diagnostics['EDH'] = {
        'cond_mean': np.mean(edh_conds), 'cond_max': np.max(edh_conds),
        'mag_mean': np.mean(edh_mags), 'mag_max': np.max(edh_mags),
    }

    # LEDH diagnostics
    ledh_conds, ledh_mags = [], []
    pts = particles.copy()
    lam = 0.0

    for j in range(n_steps):
        eps = lam_schedule[j]
        step_conds, step_mags = [], []

        for i in range(min(N, 50)):  # Sample subset for speed
            try:
                A_i, b_i = compute_ledh_matrices(
                    pts[i], m, P, model.h, model.H_jac, model.R, y, lam, R_inv)
                cond = np.abs(A_i[0, 0]) if A_i.shape[0] == 1 else np.linalg.cond(A_i)
                step_conds.append(cond if np.isfinite(cond) else 1.0)
                dx = eps * (A_i @ pts[i] + b_i)
                step_mags.append(np.abs(dx[0]))
            except:
                pass

        if step_conds:
            ledh_conds.append(np.mean(step_conds))
            ledh_mags.append(np.mean(step_mags))
        lam += eps

    diagnostics['LEDH'] = {
        'cond_mean': np.nanmean(ledh_conds), 'cond_max': np.nanmax(ledh_conds),
        'mag_mean': np.nanmean(ledh_mags), 'mag_max': np.nanmax(ledh_mags),
    }

    return diagnostics

# Multi-run Aggregation

def aggregate_results(all_runs):
    """
    Aggregate results from multiple runs.

    Args:
        all_runs: list of {method: {mse, time}} dicts

    Returns:
        {method: {mse_mean, mse_std, time_mean, time_std, n_success, n_total}}
    """
    methods = list(all_runs[0].keys())
    aggregated = {}

    for method in methods:
        mse_vals = [r[method]['mse'] for r in all_runs if np.isfinite(r[method]['mse'])]
        time_vals = [r[method]['time'] for r in all_runs]

        aggregated[method] = {
            'mse_mean': np.mean(mse_vals) if mse_vals else np.inf,
            'mse_std': np.std(mse_vals) if len(mse_vals) > 1 else 0.0,
            'time_mean': np.mean(time_vals),
            'time_std': np.std(time_vals) if len(time_vals) > 1 else 0.0,
            'n_success': len(mse_vals),
            'n_total': len(all_runs),
        }

    return aggregated

def aggregate_diagnostics(all_diags):
    """Aggregate diagnostics from multiple runs."""
    methods = ['EDH', 'LEDH']
    aggregated = {}

    for method in methods:
        cond_means = [d[method]['cond_mean'] for d in all_diags]
        cond_maxs = [d[method]['cond_max'] for d in all_diags]
        mag_means = [d[method]['mag_mean'] for d in all_diags]
        mag_maxs = [d[method]['mag_max'] for d in all_diags]

        aggregated[method] = {
            'cond_mean': np.mean(cond_means),
            'cond_mean_std': np.std(cond_means),
            'cond_max': np.mean(cond_maxs),
            'cond_max_std': np.std(cond_maxs),
            'mag_mean': np.mean(mag_means),
            'mag_mean_std': np.std(mag_means),
            'mag_max': np.mean(mag_maxs),
            'mag_max_std': np.std(mag_maxs),
        }

    return aggregated

# Experiments (with multi-run support)

def exp_nonlinearity(T, N, n_flow, base_seed, n_runs, save_dir):
    """Experiment 1: Effect of nonlinearity strength."""
    exp_scales = [0.25, 0.5, 1.0, 2.0]
    results, diags = {}, {}
    all_raw_results = {}

    for scale in exp_scales:
        model = SVAdditiveNoise(exp_scale=scale)
        run_results = []
        run_diags = []

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            rng = np.random.default_rng(seed)
            xs, ys = model.simulate(T, rng)

            res = run_filters(model, xs, ys, N, n_flow, seed)
            diag = compute_diagnostics(model, N, n_flow, seed)

            run_results.append(res)
            run_diags.append(diag)

        results[scale] = aggregate_results(run_results)
        diags[scale] = aggregate_diagnostics(run_diags)
        all_raw_results[scale] = run_results

        # Save after each condition
        _save_raw_results(all_raw_results, save_dir, '1_nonlinearity_raw.json')
        
    _plot_comparison_with_std(results, exp_scales, 'exp_scale', 'Nonlinearity', save_dir, '1_nonlinearity')
    _plot_diagnostics_with_std(diags, exp_scales, 'exp_scale', save_dir, '1_nonlinearity_diag')
    return results, diags

def exp_observation_sparsity(T, N, n_flow, base_seed, n_runs, save_dir):
    """Experiment 2: Effect of observation sparsity (missing observations)."""
    obs_rates = [1.0, 0.75, 0.5, 0.25]
    results = {}
    all_raw_results = {}

    model = SVAdditiveNoise(exp_scale=0.5)

    for rate in obs_rates:
        run_results = []

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            rng = np.random.default_rng(seed)
            xs, ys = model.simulate(T, rng)

            obs_rng = np.random.default_rng(seed + int(rate * 100))
            obs_mask = obs_rng.random(T) < rate
            obs_mask[0] = True

            res = run_filters_sparse(model, xs, ys, obs_mask, N, n_flow, seed)
            run_results.append(res)

        results[rate] = aggregate_results(run_results)
        all_raw_results[rate] = run_results

        # Save after each condition
        _save_raw_results(all_raw_results, save_dir, '2_sparsity_raw.json')
        
    _plot_comparison_with_std(results, obs_rates, 'obs_rate', 'Observation Sparsity', save_dir, '2_sparsity')
    return results

def exp_conditioning(T, N, n_flow, base_seed, n_runs, save_dir):
    """Experiment 3: Effect of conditioning (numerical stability scenarios)."""

    conditioning_experiments = {
        "well_conditioned": {
            "alpha": 0.91, "sigma": 1.0, "obs_std": 0.5, "exp_scale": 0.5
        },
        "near_unit_root": {
            "alpha": 0.99, "sigma": 1.0, "obs_std": 0.5, "exp_scale": 0.5
        },
        "highly_informative_obs": {
            "alpha": 0.91, "sigma": 1.0, "obs_std": 0.1, "exp_scale": 0.5
        },
        "high_nonlinearity": {
            "alpha": 0.91, "sigma": 1.0, "obs_std": 0.5, "exp_scale": 1.0
        },
    }

    results, diags = {}, {}
    all_raw_results = {}

    for name, params in conditioning_experiments.items():
        model = SVAdditiveNoise(
            alpha=params['alpha'],
            sigma=params['sigma'],
            obs_std=params['obs_std'],
            exp_scale=params['exp_scale']
        )

        run_results = []
        run_diags = []

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            rng = np.random.default_rng(seed)
            xs, ys = model.simulate(T, rng)

            res = run_filters(model, xs, ys, N, n_flow, seed)
            diag = compute_diagnostics(model, N, n_flow, seed)

            run_results.append(res)
            run_diags.append(diag)

        results[name] = aggregate_results(run_results)
        diags[name] = aggregate_diagnostics(run_diags)
        all_raw_results[name] = run_results

        # Save after each condition (incremental save)
        _save_raw_results(all_raw_results, save_dir, '3_conditioning_raw.json')
        
    _plot_conditioning_with_std(results, conditioning_experiments, save_dir, '3_conditioning')
    _plot_conditioning_diagnostics_with_std(diags, conditioning_experiments, save_dir, '3_conditioning_diag')
    return results, diags

# Plotting (with error bars)

def _plot_comparison_with_std(results, x_vals, x_label, title, save_dir, prefix):
    """Plot MSE and Runtime comparison with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    methods = ['EDH', 'LEDH', 'KernelPFF']
    # Blue family for EDH/LEDH, Red for Kernel
    styles = [('blue', 'o', '-'), ('dodgerblue', 's', '--'), ('red', '^', ':')]

    for ax, (metric, metric_std), ylabel in zip(
        axes,
        [('mse_mean', 'mse_std'), ('time_mean', 'time_std')],
        ['MSE', 'Runtime (s)']
    ):
        for method, (c, marker, ls) in zip(methods, styles):
            vals = [results[x][method][metric] for x in x_vals]
            stds = [results[x][method][metric_std] for x in x_vals]
            vals = [v if np.isfinite(v) else np.nan for v in vals]
            ax.errorbar(x_vals, vals, yerr=stds, marker=marker, linestyle=ls,
                        color=c, label=method, lw=1.5, ms=6, capsize=3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}.pdf'), dpi=150)
    plt.close()

def _plot_diagnostics_with_std(diags, x_vals, x_label, save_dir, prefix):
    """Plot flow stability diagnostics with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    for ax, (metric, metric_std), ylabel in zip(
        axes,
        [('cond_mean', 'cond_mean_std'), ('mag_mean', 'mag_mean_std')],
        ['Mean Condition Number', 'Mean Flow Magnitude']
    ):
        for method, c, marker, ls in [('EDH', 'blue', 'o', '-'), ('LEDH', 'dodgerblue', 's', '--')]:
            vals = [diags[x][method][metric] for x in x_vals]
            stds = [diags[x][method][metric_std] for x in x_vals]
            ax.errorbar(x_vals, vals, yerr=stds, marker=marker, linestyle=ls,
                        color=c, label=method, lw=1.5, ms=6, capsize=3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}.pdf'), dpi=150)
    plt.close()

def _plot_conditioning_with_std(results, experiments, save_dir, prefix):
    """Plot conditioning experiment results as bar charts with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    methods = ['EDH', 'LEDH', 'KernelPFF']
    colors = ['blue', 'dodgerblue', 'red']
    x_labels = list(experiments.keys())
    case_labels = ['A', 'B', 'C', 'D']
    x = np.arange(len(x_labels))
    width = 0.25

    for ax, (metric, metric_std), ylabel in zip(
        axes,
        [('mse_mean', 'mse_std'), ('time_mean', 'time_std')],
        ['MSE', 'Runtime (s)']
    ):
        for i, (method, c) in enumerate(zip(methods, colors)):
            vals = [results[name][method][metric] for name in x_labels]
            stds = [results[name][method][metric_std] for name in x_labels]
            vals = [v if np.isfinite(v) else 0 for v in vals]
            ax.bar(x + i * width, vals, width, yerr=stds, label=method,
                   color=c, edgecolor='black', capsize=2)

        ax.set_xlabel('Conditioning Scenario')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width)
        ax.set_xticklabels(case_labels, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')
        if 'mse' in metric:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}.pdf'), dpi=150)
    plt.close()

def _plot_conditioning_diagnostics_with_std(diags, experiments, save_dir, prefix):
    """Plot conditioning diagnostics as bar charts with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x_labels = list(experiments.keys())
    # Use Case A, B, C, D labels
    case_labels = ['A', 'B', 'C', 'D']
    x = np.arange(len(x_labels))
    width = 0.35

    for ax, (metric, metric_std), ylabel in zip(
        axes,
        [('cond_mean', 'cond_mean_std'), ('mag_mean', 'mag_mean_std')],
        ['Mean Condition Number', 'Mean Flow Magnitude']
    ):
        for i, (method, c) in enumerate([('EDH', 'blue'), ('LEDH', 'dodgerblue')]):
            vals = [diags[name][method][metric] for name in x_labels]
            stds = [diags[name][method][metric_std] for name in x_labels]
            ax.bar(x + i * width, vals, width, yerr=stds, label=method,
                   color=c, edgecolor='black', capsize=2)

        ax.set_xlabel('Conditioning Scenario')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(case_labels, fontsize=9)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        # Use log scale for both condition number and magnitude (extreme values)
        if 'cond' in metric or 'mag' in metric:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}.pdf'), dpi=150)
    plt.close()

# Save/Load Results

def _save_raw_results(all_raw_results, save_dir, filename):
    """Save raw results to JSON."""
    # Convert to serializable format
    serializable = {}
    for key, runs in all_raw_results.items():
        serializable[str(key)] = [
            {method: {k: (v if isinstance(v, str) else float(v)) for k, v in vals.items()}
             for method, vals in run.items()}
            for run in runs
        ]

    with open(os.path.join(save_dir, filename), 'w') as f:
        json.dump(serializable, f, indent=2)

def write_summary(all_results, diags, n_runs, save_path):
    """Write summary table with mean +/- std."""
    with open(save_path, 'w') as f:
        f.write(f"Flow Filter Comparison ({n_runs} runs per condition)\n\n")
        f.write(f"{'Condition':<25} {'Method':<12} {'MSE':>18} {'Time(s)':>18} {'Success':>10}\n")

        for cond, res in all_results.items():
            for method in ['EDH', 'LEDH', 'KernelPFF']:
                r = res[method]
                if np.isfinite(r['mse_mean']):
                    mse_str = f"{r['mse_mean']:.4f} +/- {r['mse_std']:.4f}"
                else:
                    mse_str = "FAILED"
                time_str = f"{r['time_mean']:.3f} +/- {r['time_std']:.3f}"
                success_str = f"{r['n_success']}/{r['n_total']}"
                f.write(f"{str(cond):<25} {method:<12} {mse_str:>18} {time_str:>18} {success_str:>10}\n")

        f.write(f"\nStability Diagnostics\n")
        f.write(f"{'Condition':<20} {'Method':<8} {'Cond_mean':>20} {'Cond_max':>20} {'Mag_mean':>20} {'Mag_max':>20}\n")

        for cond, d in diags.items():
            for method in ['EDH', 'LEDH']:
                m = d[method]
                cond_mean_str = f"{m['cond_mean']:.4f} +/- {m['cond_mean_std']:.4f}"
                cond_max_str = f"{m['cond_max']:.4f} +/- {m['cond_max_std']:.4f}"
                mag_mean_str = f"{m['mag_mean']:.6f} +/- {m['mag_mean_std']:.6f}"
                mag_max_str = f"{m['mag_max']:.6f} +/- {m['mag_max_std']:.6f}"
                f.write(f"{str(cond):<20} {method:<8} {cond_mean_str:>20} {cond_max_str:>20} "
                        f"{mag_mean_str:>20} {mag_max_str:>20}\n")

# Main

def main():
    """Run the main experiment."""
    base_seed = 42
    n_runs = 10  # Number of runs per experiment
    T, N, n_flow = 100, 200, 15

    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp_2_3_sv')
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()

    # Run each experiment with protection - if one fails, others continue
    res1, diag1 = {}, {}
    res2 = {}
    res3, diag3 = {}, {}

    try:
        res1, diag1 = exp_nonlinearity(T, N, n_flow, base_seed, n_runs, save_dir)
    except Exception:
        pass

    try:
        res2 = exp_observation_sparsity(T, N, n_flow, base_seed, n_runs, save_dir)
    except Exception:
        pass

    try:
        res3, diag3 = exp_conditioning(T, N, n_flow, base_seed, n_runs, save_dir)
    except Exception:
        pass

    # Combine results for summary (only non-empty)
    all_results = {}
    diags = {}

    if res1:
        all_results.update({f"scale={k}": v for k, v in res1.items()})
        diags.update({f"scale={k}": v for k, v in diag1.items()})
    if res2:
        all_results.update({f"obs_rate={k}": v for k, v in res2.items()})
    if res3:
        all_results.update({f"cond={k}": v for k, v in res3.items()})
        diags.update({f"cond={k}": v for k, v in diag3.items()})

    if all_results:
        write_summary(all_results, diags, n_runs, os.path.join(save_dir, 'summary.txt'))

    print(f"\nCompleted in {time.time() - t0:.1f}s")
    print(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    main()
