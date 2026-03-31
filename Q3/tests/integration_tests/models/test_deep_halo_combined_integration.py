"""Integration tests for DeepHalo + Sparse Demand Shock model."""

import numpy as np
import pytest

from datasets.dgp import simulate_deephalo_shrinkage_dgp
from models.deep_halo_combined import BayesianDeepHalo


# Fixtures

@pytest.fixture(scope="module")
def context_sparse_data():
    """Context effect + sparse eta DGP."""
    return simulate_deephalo_shrinkage_dgp(
        T=8, J=4, N=500, dgp_type="context_sparse", seed=42)


@pytest.fixture(scope="module")
def fitted_model(context_sparse_data):
    """Fitted BayesianDeepHalo with small NN/MCMC for speed."""
    X, q, true_params = context_sparse_data
    model = BayesianDeepHalo(
        embedding_dim=8, n_layers=2, n_heads=2,
        nn_lr=1e-3, nn_epochs=20,
        n_mcmc=40, n_burnin=15,
    )
    model.fit(X, q)
    return model, X, q, true_params


class TestBayesianDeepHaloFit:
    """Fit pipeline tests."""

    def test_fit_sets_is_fitted(self, fitted_model):
        model, _, _, _ = fitted_model
        assert model.is_fitted

    def test_fit_populates_samples(self, fitted_model):
        model, _, _, _ = fitted_model
        assert model.samples_ is not None
        for key in ("eta", "gamma", "phi"):
            assert key in model.samples_

    def test_samples_shapes(self, fitted_model):
        """Samples have (n_mcmc - n_burnin, ...) shape."""
        model, _, _, _ = fitted_model
        n_eff = 40 - 15  # n_mcmc - n_burnin
        assert model.samples_["eta"].shape == (n_eff, 8, 4)
        assert model.samples_["gamma"].shape == (n_eff, 8, 4)
        assert model.samples_["phi"].shape == (n_eff, 8)

    def test_nn_is_instantiated(self, fitted_model):
        model, _, _, _ = fitted_model
        assert hasattr(model, '_nn')
        assert len(model._nn.trainable_weights) > 0

    def test_h_fixed_shape(self, fitted_model):
        model, _, _, _ = fitted_model
        assert model._h_fixed.shape == (8, 4)


class TestBayesianDeepHaloPredict:
    """Prediction tests."""

    def test_compute_shares_shape(self, fitted_model):
        model, X, _, _ = fitted_model
        shares = model.compute_shares(X)
        assert shares.shape == (8, 5)  # (T, J+1)

    def test_compute_shares_sum_to_one(self, fitted_model):
        model, X, _, _ = fitted_model
        shares = model.compute_shares(X)
        assert np.allclose(np.sum(shares, axis=1), 1.0, atol=1e-5)

    def test_compute_shares_positive(self, fitted_model):
        model, X, _, _ = fitted_model
        shares = model.compute_shares(X)
        assert np.all(shares > 0)

    def test_prediction_consistency(self, fitted_model):
        """Same call twice yields same result."""
        model, X, _, _ = fitted_model
        s1 = model.compute_shares(X)
        s2 = model.compute_shares(X)
        assert np.allclose(s1, s2)


class TestBayesianDeepHaloSummary:
    """Posterior and sparsity summary tests."""

    def test_posterior_summary_keys(self, fitted_model):
        model, _, _, _ = fitted_model
        summary = model.get_posterior_summary()
        assert "eta" in summary
        for key in ("mean", "std", "ci_lower", "ci_upper"):
            assert key in summary["eta"]

    def test_posterior_summary_eta_shape(self, fitted_model):
        model, _, _, _ = fitted_model
        summary = model.get_posterior_summary()
        assert summary["eta"]["mean"].shape == (8, 4)

    def test_sparsity_summary_keys(self, fitted_model):
        model, _, _, _ = fitted_model
        sparsity = model.get_sparsity_summary()
        assert "prob_gamma_1" in sparsity
        assert "frac_active" in sparsity

    def test_sparsity_summary_values(self, fitted_model):
        model, _, _, _ = fitted_model
        sparsity = model.get_sparsity_summary()
        assert sparsity["prob_gamma_1"].shape == (8, 4)
        assert np.all(sparsity["prob_gamma_1"] >= 0.0)
        assert np.all(sparsity["prob_gamma_1"] <= 1.0)
        assert 0.0 <= sparsity["frac_active"] <= 1.0


class TestBayesianDeepHaloRefit:
    """Refit shocks tests."""

    def test_refit_returns_shares_and_samples(self, fitted_model, context_sparse_data):
        """refit_shocks returns correct types and shapes."""
        model, _, _, _ = fitted_model
        X_new, q_new, _ = context_sparse_data  # reuse same data for simplicity
        # Use very small MCMC for refit speed
        new_shares, new_samples, new_chain = model.refit_shocks(
            X_new, q_new, n_mcmc=20, n_burnin=5)

        T_new, J = X_new.shape[0], X_new.shape[1]
        assert new_shares.shape == (T_new, J + 1)
        assert np.allclose(np.sum(new_shares, axis=1), 1.0, atol=1e-5)
        for key in ("eta", "gamma", "phi"):
            assert key in new_samples

    def test_refit_preserves_original_state(self, fitted_model, context_sparse_data):
        """After refit, original model state is restored."""
        model, X, q, _ = fitted_model
        orig_T = model.T
        orig_fitted = model.is_fitted
        orig_eta_shape = model.samples_["eta"].shape

        X_new, q_new, _ = context_sparse_data
        model.refit_shocks(X_new, q_new, n_mcmc=20, n_burnin=5)

        assert model.T == orig_T
        assert model.is_fitted == orig_fitted
        assert model.samples_["eta"].shape == orig_eta_shape


class TestBayesianDeepHaloRobustness:
    """Robustness tests."""

    def test_fit_returns_self(self, context_sparse_data):
        """fit() returns the model for chaining."""
        X, q, _ = context_sparse_data
        model = BayesianDeepHalo(
            embedding_dim=8, n_layers=2, n_heads=2,
            nn_epochs=5, n_mcmc=15, n_burnin=5,
        )
        result = model.fit(X, q)
        assert result is model


@pytest.mark.parametrize("dgp_type", ["context_sparse", "nocontext_sparse"])
def test_fit_different_dgps(dgp_type):
    """Both context/no-context DGPs produce valid fits."""
    X, q, _ = simulate_deephalo_shrinkage_dgp(
        T=6, J=3, N=300, dgp_type=dgp_type, seed=42)
    model = BayesianDeepHalo(
        embedding_dim=8, n_layers=2, n_heads=2,
        nn_epochs=10, n_mcmc=20, n_burnin=5,
    )
    model.fit(X, q)

    assert model.is_fitted
    shares = model.compute_shares(X)
    assert np.allclose(np.sum(shares, axis=1), 1.0, atol=1e-5)
