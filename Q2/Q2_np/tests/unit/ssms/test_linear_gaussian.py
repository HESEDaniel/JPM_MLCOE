"""Unit tests for Linear Gaussian SSM."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ssm import linear_gaussian_ssm


class TestLinearGaussianSSM:
    """Unit tests for Linear Gaussian SSM."""

    def test_output_shapes(self, rng, kf_system):
        """Generated data should have correct shapes."""
        A, B, C, D, Sigma = kf_system
        T = 100

        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

        assert xs.shape == (T, 2), f"xs shape: {xs.shape}"
        assert ys.shape == (T, 1), f"ys shape: {ys.shape}"

    def test_no_nan(self, rng, kf_system):
        """Generated data should not contain NaN."""
        A, B, C, D, Sigma = kf_system
        T = 100

        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_reproducibility(self, kf_system):
        """Same seed should give same results."""
        A, B, C, D, Sigma = kf_system
        T = 50

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        xs1, ys1 = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng1)
        xs2, ys2 = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng2)

        np.testing.assert_array_equal(xs1, xs2)
        np.testing.assert_array_equal(ys1, ys2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
