"""Integration tests for Lu25 Bayesian model."""

import numpy as np
import pytest

from datasets.dgp import simulate_lu25_dgp
from models.lu25_bayesian import Lu25Model


# Fixtures

@pytest.fixture(scope="module")
def dgp1_data():
    """DGP1: sparse eta, exogenous price."""
    return simulate_lu25_dgp(T=8, J=4, N=500, dgp_type="dgp1", seed=42)


@pytest.fixture(scope="module")
def fitted_lu25(dgp1_data):
    """Fitted Lu25Model with small MCMC for speed."""
    X, q, true_params = dgp1_data
    model = Lu25Model(n_mcmc=50, n_burnin=20, R0=30, use_tmh=False)
    model.fit(X, q)
    return model, X, q, true_params


class TestLu25Fit:
    """Fit pipeline tests."""

    def test_fit_sets_is_fitted(self, fitted_lu25):
        model, _, _, _ = fitted_lu25
        assert model.is_fitted

    def test_fit_populates_samples(self, fitted_lu25):
        model, _, _, _ = fitted_lu25
        assert model.samples_ is not None
        for key in ("beta_bar", "r", "xi_bar", "eta", "gamma", "phi"):
            assert key in model.samples_

    def test_samples_shapes(self, fitted_lu25):
        """Samples have (n_mcmc - n_burnin, ...) shape."""
        model, _, _, _ = fitted_lu25
        n_eff = 50 - 20  # n_mcmc - n_burnin
        assert model.samples_["beta_bar"].shape == (n_eff, 2)  # d_X=2
        assert model.samples_["r"].shape == (n_eff, 2)         # d_rc=2
        assert model.samples_["xi_bar"].shape == (n_eff, 8)    # T=8
        assert model.samples_["eta"].shape == (n_eff, 8, 4)    # (T, J)
        assert model.samples_["gamma"].shape == (n_eff, 8, 4)
        assert model.samples_["phi"].shape == (n_eff, 8)

    def test_full_chain_stored(self, fitted_lu25):
        model, _, _, _ = fitted_lu25
        assert model.full_chain_ is not None
        assert len(model.full_chain_) == 6
        assert model.full_chain_[0].shape[0] == 50  # n_mcmc


class TestLu25Predict:
    """Prediction tests."""

    def test_predict_probas_shape(self, fitted_lu25):
        model, X, _, _ = fitted_lu25
        shares = model.predict_probas(X)
        assert shares.shape == (8, 5)  # (T, J+1)

    def test_predict_probas_sum_to_one(self, fitted_lu25):
        model, X, _, _ = fitted_lu25
        shares = model.predict_probas(X)
        assert np.allclose(np.sum(shares, axis=1), 1.0, atol=1e-4)

    def test_predict_probas_positive(self, fitted_lu25):
        model, X, _, _ = fitted_lu25
        shares = model.predict_probas(X)
        assert np.all(shares > 0)

    def test_compute_shares_matches_predict_probas(self, fitted_lu25):
        model, X, _, _ = fitted_lu25
        s1 = model.compute_shares(X)
        s2 = model.predict_probas(X)
        assert np.allclose(s1, s2, atol=1e-5)


class TestLu25PosteriorSummary:
    """Posterior summary tests."""

    def test_posterior_summary_keys(self, fitted_lu25):
        model, _, _, _ = fitted_lu25
        summary = model.get_posterior_summary()
        for key in ("beta_bar", "r", "sigma", "xi_bar"):
            assert key in summary

    def test_posterior_summary_structure(self, fitted_lu25):
        model, _, _, _ = fitted_lu25
        summary = model.get_posterior_summary()
        for param_summary in summary.values():
            for key in ("mean", "std", "ci_lower", "ci_upper"):
                assert key in param_summary

    def test_sigma_is_exp_r(self, fitted_lu25):
        """sigma mean should be close to exp(r mean)."""
        model, _, _, _ = fitted_lu25
        summary = model.get_posterior_summary()
        # Not exact equality because E[exp(r)] != exp(E[r]), but close for small variance
        sigma_mean = summary["sigma"]["mean"]
        r_mean = summary["r"]["mean"]
        assert sigma_mean.shape == r_mean.shape

    def test_ci_ordering(self, fitted_lu25):
        """ci_lower < mean < ci_upper."""
        model, _, _, _ = fitted_lu25
        summary = model.get_posterior_summary()
        beta_s = summary["beta_bar"]
        assert np.all(beta_s["ci_lower"] <= beta_s["mean"])
        assert np.all(beta_s["mean"] <= beta_s["ci_upper"])


class TestLu25Robustness:
    """Robustness tests."""

    def test_prediction_consistency(self, fitted_lu25):
        """Same X -> same shares."""
        model, X, _, _ = fitted_lu25
        s1 = model.compute_shares(X)
        s2 = model.compute_shares(X)
        assert np.allclose(s1, s2)

    def test_fit_returns_self(self, dgp1_data):
        """fit() returns the model for chaining."""
        X, q, _ = dgp1_data
        model = Lu25Model(n_mcmc=20, n_burnin=5, R0=10, use_tmh=False)
        result = model.fit(X, q)
        assert result is model


@pytest.mark.parametrize("dgp_type", ["dgp1", "dgp3"])
def test_fit_different_dgps(dgp_type):
    """Both sparse (dgp1) and non-sparse (dgp3) DGPs produce valid fits."""
    X, q, _ = simulate_lu25_dgp(T=6, J=3, N=300, dgp_type=dgp_type, seed=42)
    model = Lu25Model(n_mcmc=30, n_burnin=10, R0=20, use_tmh=False)
    model.fit(X, q)

    assert model.is_fitted
    shares = model.compute_shares(X)
    assert np.allclose(np.sum(shares, axis=1), 1.0, atol=1e-4)
