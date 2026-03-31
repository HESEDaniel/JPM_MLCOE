"""Root conftest.py - Shared pytest fixtures for TF tests."""
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
