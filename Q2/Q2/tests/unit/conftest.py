"""Shared pytest fixtures for TF unit tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

DTYPE = tf.float64


@pytest.fixture
def rng():
    """Seeded numpy RNG for data generation."""
    return np.random.default_rng(42)


@pytest.fixture
def tf_rng():
    """Seeded TF RNG for filters/flows."""
    return tf.random.Generator.from_seed(42)


@pytest.fixture
def kf_system():
    """KF-specific system matrices (A, B, C, D, Sigma)."""
    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.1]])
    Sigma = np.eye(2)
    return A, B, C, D, Sigma


@pytest.fixture
def linear_model():
    """Simple 2D linear model for testing (TF SSM object)."""
    from src.ssm import LinearGaussianSSM

    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.eye(2)
    D = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma = np.eye(2)

    return LinearGaussianSSM(A, B, C, D, Sigma)


@pytest.fixture
def linear_ssm_data(rng, linear_model):
    """Generate data from the linear SSM."""
    T = 30
    xs, ys = linear_model.simulate(T, rng)
    ys_tf = tf.constant(ys, dtype=DTYPE)
    return {
        'ssm': linear_model,
        'xs': xs,
        'ys': ys,
        'ys_tf': ys_tf,
        'T': T,
    }


@pytest.fixture
def nonlinear_model():
    """Nonlinear range-bearing-like model using a simple 2D SSM."""
    from src.ssm import LinearGaussianSSM

    # Use a simple 2D system but we'll test with EKF/UKF which work on it
    A = np.eye(2) * 0.99
    B = np.sqrt(0.1) * np.eye(2)
    C = np.eye(2)
    D = np.sqrt(0.1) * np.eye(2)
    Sigma = np.array([[0.5, 0.1], [0.1, 0.5]])

    return LinearGaussianSSM(A, B, C, D, Sigma)


def check_psd(matrix_np, tol=1e-10):
    """Check if a numpy matrix is positive semi-definite."""
    eigvals = np.linalg.eigvalsh(matrix_np)
    return np.all(eigvals >= -tol)
