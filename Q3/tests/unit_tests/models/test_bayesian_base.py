"""Unit tests for BayesianChoiceModel base class."""

import numpy as np
import pytest

from models.bayesian_base_model import BayesianChoiceModel


# --- Minimal concrete subclass for testing ---

class _DummyBayesianModel(BayesianChoiceModel):
    """Minimal concrete subclass for testing base class logic."""

    def _prepare_data(self, X, q):
        self.X, self.q = X, q

    def _init_state(self):
        self._init_state_tuple = (np.zeros(2),)

    def _make_kernel(self):
        pass

    def _get_kappa_blocks(self, kernel):
        return []

    def _extract_samples(self, chain, trace):
        self.samples_ = {"theta": np.random.randn(10, 2)}

    def compute_shares(self, X):
        return np.ones((2, 3)) / 3


# Fixtures

@pytest.fixture
def unfitted_model():
    return _DummyBayesianModel()


class TestBayesianChoiceModelInit:
    """Instantiation and default parameter tests."""

    def test_default_parameters(self, unfitted_model):
        """Test default MCMC parameters."""
        assert unfitted_model.n_mcmc == 5000
        assert unfitted_model.n_burnin == 1500
        assert unfitted_model.R0 == 200

    def test_custom_parameters(self):
        """Test custom MCMC parameters are stored."""
        model = _DummyBayesianModel(n_mcmc=100, n_burnin=10, R0=50)
        assert model.n_mcmc == 100
        assert model.n_burnin == 10
        assert model.R0 == 50

    def test_initial_state(self, unfitted_model):
        """Test model starts unfitted with no samples."""
        assert not unfitted_model.is_fitted
        assert unfitted_model.samples_ is None


class TestBayesianChoiceModelGuards:
    """Error handling for unfitted model."""

    def test_predict_probas_before_fit(self, unfitted_model):
        """predict_probas raises RuntimeError before fit."""
        with pytest.raises(RuntimeError, match="Call fit"):
            unfitted_model.predict_probas(None)

    def test_get_posterior_summary_before_fit(self, unfitted_model):
        """get_posterior_summary raises RuntimeError before fit."""
        with pytest.raises(RuntimeError, match="Call fit"):
            unfitted_model.get_posterior_summary()


class TestBayesianChoiceModelPosteriorSummary:
    """Posterior summary computation tests."""

    def test_summary_structure(self):
        """Summary dict has correct keys and sub-keys."""
        model = _DummyBayesianModel()
        model.is_fitted = True
        model.samples_ = {"beta": np.random.randn(100, 3)}

        summary = model.get_posterior_summary()
        assert "beta" in summary
        for key in ("mean", "std", "ci_lower", "ci_upper"):
            assert key in summary["beta"]

    def test_summary_values(self):
        """Summary values are correct for known distribution."""
        model = _DummyBayesianModel()
        model.is_fitted = True
        np.random.seed(42)
        model.samples_ = {"theta": np.random.normal(2.0, 0.5, size=(5000, 1))}

        summary = model.get_posterior_summary()["theta"]
        assert np.isclose(summary["mean"][0], 2.0, atol=0.05)
        assert summary["ci_lower"][0] < 2.0 < summary["ci_upper"][0]

    def test_summary_shape(self):
        """Summary fields have correct shape matching parameter dimension."""
        model = _DummyBayesianModel()
        model.is_fitted = True
        model.samples_ = {"beta": np.random.randn(200, 4)}

        summary = model.get_posterior_summary()["beta"]
        for key in ("mean", "std", "ci_lower", "ci_upper"):
            assert summary[key].shape == (4,)
