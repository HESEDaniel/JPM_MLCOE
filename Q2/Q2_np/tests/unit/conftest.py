"""Shared pytest fixtures for unit tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def linear_model():
    """Simple 2D linear model for testing."""
    n_x = 2
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    H = np.eye(n_x)
    Q = 0.1 * np.eye(n_x)
    R = 0.1 * np.eye(n_x)
    m0 = np.zeros(n_x)
    P0 = np.eye(n_x)

    def f(x):
        return (A @ x.T).T if x.ndim > 1 else A @ x

    def h(x):
        return (H @ x.T).T if x.ndim > 1 else H @ x

    def F_jac(x):
        return A

    def H_jac(x):
        return H

    return {
        'f': f, 'h': h, 'F_jac': F_jac, 'H_jac': H_jac,
        'A': A, 'H': H, 'Q': Q, 'R': R, 'm0': m0, 'P0': P0
    }


@pytest.fixture
def linear_ssm(rng, linear_model):
    """Generate linear SSM data."""
    T, n_x, n_y = 30, 2, 2
    m = linear_model

    xs = np.zeros((T, n_x))
    ys = np.zeros((T, n_y))
    x = rng.multivariate_normal(m['m0'], m['P0'])

    for t in range(T):
        x = m['f'](x) + rng.multivariate_normal(np.zeros(n_x), m['Q'])
        y = m['h'](x) + rng.multivariate_normal(np.zeros(n_y), m['R'])
        xs[t], ys[t] = x, y

    return {**m, 'xs': xs, 'ys': ys, 'T': T}


@pytest.fixture
def nonlinear_model():
    """Nonlinear range-bearing model for testing."""
    n_x = 2
    A = np.eye(n_x) * 0.99
    Q = 0.1 * np.eye(n_x)
    R = np.diag([0.1, 0.05])
    m0 = np.array([3.0, 3.0])
    P0 = np.array([[0.5, 0.1], [0.1, 0.5]])
    y = np.array([4.2, 0.7])

    def f(x):
        return (A @ x.T).T if x.ndim > 1 else A @ x

    def F_jac(x):
        return A

    def h(x):
        if x.ndim > 1:
            return np.array([h(xi) for xi in x])
        r = np.sqrt(x[0]**2 + x[1]**2)
        theta = np.arctan2(x[1], x[0])
        return np.array([r, theta])

    def H_jac(x):
        r = max(np.sqrt(x[0]**2 + x[1]**2), 1e-6)
        return np.array([[x[0]/r, x[1]/r], [-x[1]/r**2, x[0]/r**2]])

    return {
        'f': f, 'h': h, 'F_jac': F_jac, 'H_jac': H_jac,
        'A': A, 'Q': Q, 'R': R, 'm0': m0, 'P0': P0, 'y': y
    }


@pytest.fixture
def kf_system():
    """KF-specific system (uses A, B, C, D matrices)."""
    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0], [0, 0.1]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.1]])
    Sigma = np.eye(2)
    return A, B, C, D, Sigma


def check_psd(matrix, tol=1e-10):
    """Check if matrix is positive semi-definite."""
    return np.all(np.linalg.eigvalsh(matrix) >= -tol)
