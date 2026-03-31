"""Kalman Filter numerical stability comparison."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm import linear_gaussian_ssm
from src.filters import kalman_filter
from src.utils.metrics import compute_mse

METHODS = [
    ('cholesky', False, 'Standard'),
    ('cholesky', True, 'Joseph'),
]

def get_normal_systems():
    """Normal test systems with varying dimensions."""
    return {
        '1d': {
            'A': np.array([[0.95]]),
            'B': np.array([[0.5]]),
            'C': np.array([[1.0]]),
            'D': np.array([[0.3]]),  # R = 0.09
            'Sigma': np.array([[1.0]]),
        },
        '2d': {
            'A': np.array([[1.0, 1.0],
                          [0.0, 1.0]]),  # Random walk with velocity
            'B': np.array([[0.5, 0.0],
                          [0.0, 0.5]]),  # Q = 0.25 * I
            'C': np.array([[1.0, 0.0]]),  # Observe position only
            'D': np.array([[0.1]]),  # R = 0.01
            'Sigma': np.array([[1.0, 0.0],
                              [0.0, 1.0]]),
        },
        '10d': {
            # Near-identity dynamics with coupling
            'A': 0.95 * np.eye(10) + 0.02 * np.diag(np.ones(9), 1),
            'B': 0.3 * np.eye(10),  # Q = 0.09 * I
            'C': np.eye(5, 10),  # Observe first 5 states only
            'D': 0.2 * np.eye(5),  # R = 0.04 * I
            'Sigma': np.eye(10),
        },
    }

def get_high_precision_systems():
    """1D systems with decreasing observation noise (high precision)."""
    return {
        'D=1e-6': {
            'A': np.array([[1.0]]),
            'B': np.array([[1.0]]),
            'C': np.array([[1.0]]),
            'D': np.array([[1e-6]]),  # R = 1e-12
            'Sigma': np.array([[1.0]]),
        },
        'D=1e-7': {
            'A': np.array([[1.0]]),
            'B': np.array([[1.0]]),
            'C': np.array([[1.0]]),
            'D': np.array([[1e-7]]),  # R = 1e-14
            'Sigma': np.array([[1.0]]),
        },
        'D=1e-8': {
            'A': np.array([[1.0]]),
            'B': np.array([[1.0]]),
            'C': np.array([[1.0]]),
            'D': np.array([[1e-8]]),  # R = 1e-16 -> K ~ 1 -> P collapses
            'Sigma': np.array([[1.0]]),
        },
    }

def run_method(params, ys, xs, solver, joseph):
    """Run KF and return metrics: mse, cond, runtime."""
    A, B, C, D, Sigma = params['A'], params['B'], params['C'], params['D'], params['Sigma']
    result = {'failed': False}
    try:
        t0 = time.perf_counter()
        m, P, cond = kalman_filter(A, B, C, D, Sigma, ys, joseph=joseph, solver=solver)
        runtime = time.perf_counter() - t0

        max_cond = np.max(cond)

        # Check failure: NaN, Inf, or kappa=inf (P collapsed)
        if np.any(np.isnan(P)) or np.any(np.isinf(P)) or np.isinf(max_cond):
            result['failed'], result['reason'] = True, 'P collapsed (kappa=inf)'
            return result

        result.update({
            'm': m, 'P': P,
            'mse': compute_mse(m, xs),
            'cond': max_cond,
            'runtime': runtime * 1000,  # ms
        })
    except np.linalg.LinAlgError as e:
        result['failed'], result['reason'] = True, str(e)
    return result

def run_all(systems, T, seed):
    """Run all methods on all systems."""
    results = {}
    for name, params in systems.items():
        rng = np.random.default_rng(seed)
        xs, ys = linear_gaussian_ssm(**params, T=T, rng=rng)
        results[name] = {'xs': xs, 'ys': ys}
        for solver, joseph, label in METHODS:
            results[name][label] = run_method(params, ys, xs, solver, joseph)
    return results

def print_table(results, title):
    """Print a single results table."""
    for name, res in results.items():
        for _, _, label in METHODS:
            r = res[label]
            if r['failed']:
                print(f"{name:<12} {label:<10} {'---':<12} {'inf':<14} {'---':<12} FAIL")
            else:
                log_cond = np.log10(r['cond']) if r['cond'] > 0 else 0

def save_report(results_normal, results_highprec, filepath):
    """Save results to text file with two tables."""
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("KALMAN FILTER NUMERICAL STABILITY: Standard vs Joseph Update\n")
        f.write("="*70 + "\n\n")

        # Table 1: Normal systems (varying dimension)
        f.write("TABLE 1: Varying State Dimension (Normal Noise)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'System':<12} {'Method':<10} {'MSE':<12} {'log10(kappa)':<14} {'Runtime(ms)':<12} {'Status'}\n")
        f.write("-"*70 + "\n")
        for name, res in results_normal.items():
            for _, _, label in METHODS:
                r = res[label]
                if r['failed']:
                    f.write(f"{name:<12} {label:<10} {'---':<12} {'inf':<14} {'---':<12} FAIL\n")
                else:
                    log_cond = np.log10(r['cond']) if r['cond'] > 0 else 0
                    f.write(f"{name:<12} {label:<10} {r['mse']:<12.6f} {log_cond:<14.2f} {r['runtime']:<12.2f} OK\n")

        f.write("\n\n")

        # Table 2: High-precision systems
        f.write("TABLE 2: High Precision Observations (1d, varying R)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'System':<12} {'Method':<10} {'MSE':<12} {'log10(kappa)':<14} {'Runtime(ms)':<12} {'Status'}\n")
        f.write("-"*70 + "\n")
        for name, res in results_highprec.items():
            for _, _, label in METHODS:
                r = res[label]
                if r['failed']:
                    f.write(f"{name:<12} {label:<10} {'---':<12} {'inf':<14} {'---':<12} FAIL\n")
                else:
                    log_cond = np.log10(r['cond']) if r['cond'] > 0 else 0
                    f.write(f"{name:<12} {label:<10} {r['mse']:<12.6f} {log_cond:<14.2f} {r['runtime']:<12.2f} OK\n")

    print(f"Report saved: {filepath}")

if __name__ == "__main__":
    save_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'exp_1_1_kalman')
    os.makedirs(save_path, exist_ok=True)

    # Run both experiment groups
    results_normal = run_all(get_normal_systems(), T=200, seed=42)
    results_highprec = run_all(get_high_precision_systems(), T=200, seed=42)

    # Print tables
    print_table(results_normal, "TABLE 1: Varying State Dimension (Normal Noise)")
    print_table(results_highprec, "TABLE 2: High Precision Observations (1d, varying R)")

    # Save report
    save_report(results_normal, results_highprec, os.path.join(save_path, 'stability_report.txt'))
