"""Root conftest.py - Shared pytest fixtures for all tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)
