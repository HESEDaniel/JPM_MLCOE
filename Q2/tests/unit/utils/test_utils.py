"""Unit tests for numerical utility functions."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.utils import exponential_lambda_schedule


class TestExponentialLambdaSchedule:
    """Tests for exponential lambda schedule generation."""

    def test_output_shape(self):
        """Schedule should have n_steps + 1 positions."""
        n_steps = 29

        lam = exponential_lambda_schedule(n_steps)

        assert lam.shape == (n_steps + 1,)

    def test_endpoints(self):
        """Schedule should start at 0 and end at 1."""
        n_steps = 29

        lam = exponential_lambda_schedule(n_steps)

        np.testing.assert_allclose(lam[0], 0.0)
        np.testing.assert_allclose(lam[-1], 1.0)

    def test_default_parameters(self):
        """Default parameters should match Li et al. 2017."""
        lam = exponential_lambda_schedule()  # n_steps=29, ratio=1.2

        assert len(lam) == 30  # 29 steps + 1
        np.testing.assert_allclose(lam[0], 0.0)
        np.testing.assert_allclose(lam[-1], 1.0)

    @pytest.mark.parametrize("n_steps", [5, 10, 50])
    def test_different_n_steps(self, n_steps):
        """Should work for different numbers of steps."""
        lam = exponential_lambda_schedule(n_steps)

        assert len(lam) == n_steps + 1
        np.testing.assert_allclose(lam[0], 0.0)
        np.testing.assert_allclose(lam[-1], 1.0)
        assert np.all(np.diff(lam) > 0)  # Increasing

    @pytest.mark.parametrize("ratio", [1.1, 1.5, 2.0])
    def test_different_ratios(self, ratio):
        """Should work for different ratios."""
        n_steps = 20
        lam = exponential_lambda_schedule(n_steps, ratio=ratio)

        assert len(lam) == n_steps + 1
        np.testing.assert_allclose(lam[0], 0.0)
        np.testing.assert_allclose(lam[-1], 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
