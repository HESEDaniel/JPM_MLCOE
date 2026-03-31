"""Integration tests for PyBLP estimator."""

import numpy as np
import pytest

from choice_learn.datasets.dgp import simulate_lu25_dgp
from choice_learn.models.pyblp import PyBLPEstimator


# Fixtures

@pytest.fixture(scope="module")
def dgp2_data():
    """DGP2: sparse eta, endogenous price (appropriate for cost_iv)."""
    return simulate_lu25_dgp(T=10, J=5, N=500, dgp_type="dgp2", seed=42)


@pytest.fixture(scope="module")
def dgp1_data():
    """DGP1: sparse eta, exogenous price (for weak_iv)."""
    return simulate_lu25_dgp(T=10, J=5, N=500, dgp_type="dgp1", seed=42)


@pytest.fixture(scope="module")
def fitted_cost_iv(dgp2_data):
    """Fitted BLP with cost IV."""
    X, q, true_params = dgp2_data
    est = PyBLPEstimator(iv_type="cost_iv", n_draws=200, seed=42)
    est.fit(X, q, cost_shock=true_params["cost_shock"])
    return est, true_params


class TestPyBLPCostIV:
    """Cost IV estimation tests."""

    def test_fit_cost_iv(self, fitted_cost_iv):
        est, _ = fitted_cost_iv
        assert est.is_fitted

    def test_get_results_keys(self, fitted_cost_iv):
        est, _ = fitted_cost_iv
        results = est.get_results()
        for key in ("xi_bar_est", "beta_p_est", "beta_w_est", "sigma_est"):
            assert key in results

    def test_get_results_with_true_params(self, fitted_cost_iv):
        est, true_params = fitted_cost_iv
        results = est.get_results(true_params)
        for key in ("beta_p_bias", "beta_w_bias", "sigma_bias",
                     "xi_bar_bias", "xi_jt_abs_bias", "xi_jt_sd"):
            assert key in results

    def test_estimates_finite(self, fitted_cost_iv):
        est, _ = fitted_cost_iv
        results = est.get_results()
        for val in results.values():
            assert np.isfinite(val)

    def test_sigma_positive(self, fitted_cost_iv):
        est, _ = fitted_cost_iv
        assert est.sigma_hat > 0


class TestPyBLPWeakIV:
    """Weak IV estimation tests."""

    def test_fit_weak_iv(self, dgp1_data):
        X, q, _ = dgp1_data
        est = PyBLPEstimator(iv_type="weak_iv", n_draws=200, seed=42)
        est.fit(X, q)
        assert est.is_fitted

    def test_get_results_keys(self, dgp1_data):
        X, q, _ = dgp1_data
        est = PyBLPEstimator(iv_type="weak_iv", n_draws=200, seed=42)
        est.fit(X, q)
        results = est.get_results()
        for key in ("xi_bar_est", "beta_p_est", "beta_w_est", "sigma_est"):
            assert key in results


class TestPyBLPRobustness:
    """Robustness tests."""

    def test_fit_returns_self(self, dgp2_data):
        X, q, true_params = dgp2_data
        est = PyBLPEstimator(iv_type="cost_iv", n_draws=200, seed=42)
        result = est.fit(X, q, cost_shock=true_params["cost_shock"])
        assert result is est

    def test_stored_attributes(self, fitted_cost_iv):
        est, _ = fitted_cost_iv
        assert est.T == 10
        assert est.J == 5
        assert est.xi_hat.shape == (10, 5)
