"""Unit tests for DeepHalo + Sparse Demand Shock model."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.models.deep_halo_combined import (
    BayesianDeepHalo,
    ShockKernelResults,
    SparseDemandShockKernel,
    _logit_shares,
)


# --- Logit share helper ---

class TestLogitShares:
    """Tests for _logit_shares."""

    def test_output_shape(self):
        """v (T,J) -> (T,J+1)."""
        v = tf.constant(np.random.randn(5, 3).astype(np.float32))
        shares = _logit_shares(v)
        assert shares.shape == (5, 4)

    def test_sums_to_one(self):
        """Each row sums to 1."""
        v = tf.constant(np.random.randn(5, 3).astype(np.float32))
        shares = _logit_shares(v).numpy()
        assert np.allclose(np.sum(shares, axis=1), 1.0, atol=1e-6)

    def test_all_positive(self):
        """All shares positive."""
        v = tf.constant(np.random.randn(5, 3).astype(np.float32))
        shares = _logit_shares(v).numpy()
        assert np.all(shares > 0)

    def test_zero_input_uniform(self):
        """All-zero v -> uniform shares over J+1."""
        v = tf.zeros((3, 4))
        shares = _logit_shares(v).numpy()
        assert np.allclose(shares, 1.0 / 5, atol=1e-6)

    @pytest.mark.parametrize("T,J", [(1, 1), (1, 5), (10, 3)])
    def test_output_shape_parametrized(self, T, J):
        """Various input sizes produce correct output shape."""
        v = tf.random.normal((T, J))
        shares = _logit_shares(v)
        assert shares.shape == (T, J + 1)


# --- SparseDemandShockKernel ---

class TestSparseDemandShockKernel:
    """Tests for the shock MCMC kernel."""

    @pytest.fixture
    def kernel_and_state(self):
        """Create kernel and initial state."""
        T, J = 5, 3
        h_fixed = tf.constant(np.random.randn(T, J).astype(np.float32))
        q = tf.constant(np.abs(np.random.randn(T, J + 1)).astype(np.float32))
        kernel = SparseDemandShockKernel(
            h_fixed, q, kappa_eta=0.1,
            tau0_sq=1e-3, tau1_sq=1.0, a_phi=1.0, b_phi=1.0,
        )
        state = (
            tf.zeros((T, J)),        # eta
            tf.zeros((T, J)),        # gamma
            tf.fill((T,), 0.1),      # phi
        )
        return kernel, state, T, J

    def test_is_calibrated(self, kernel_and_state):
        kernel, _, _, _ = kernel_and_state
        assert kernel.is_calibrated

    def test_bootstrap_results(self, kernel_and_state):
        kernel, state, _, _ = kernel_and_state
        kr = kernel.bootstrap_results(state)
        assert isinstance(kr, ShockKernelResults)
        assert float(kr.acc_eta) == 0.0

    def test_one_step_output_shapes(self, kernel_and_state):
        """State shapes preserved after one step."""
        kernel, state, T, J = kernel_and_state
        kr = kernel.bootstrap_results(state)
        new_state, new_kr = kernel.one_step(state, kr)

        assert new_state[0].shape == (T, J)   # eta
        assert new_state[1].shape == (T, J)   # gamma
        assert new_state[2].shape == (T,)     # phi

    def test_one_step_returns_kernel_results(self, kernel_and_state):
        kernel, state, _, _ = kernel_and_state
        kr = kernel.bootstrap_results(state)
        _, new_kr = kernel.one_step(state, kr)
        assert isinstance(new_kr, ShockKernelResults)

    def test_gamma_binary_after_step(self, kernel_and_state):
        """gamma values are 0 or 1 after one step."""
        kernel, state, _, _ = kernel_and_state
        kr = kernel.bootstrap_results(state)
        new_state, _ = kernel.one_step(state, kr)
        gamma = new_state[1].numpy()
        assert np.all((gamma == 0.0) | (gamma == 1.0))

    def test_phi_in_unit_interval(self, kernel_and_state):
        """phi values in [0, 1] after one step."""
        kernel, state, _, _ = kernel_and_state
        kr = kernel.bootstrap_results(state)
        new_state, _ = kernel.one_step(state, kr)
        phi = new_state[2].numpy()
        assert np.all(phi >= 0.0)
        assert np.all(phi <= 1.0)


# --- BayesianDeepHalo ---

class TestBayesianDeepHalo:
    """Tests for BayesianDeepHalo initialization and guards."""

    def test_default_parameters(self):
        model = BayesianDeepHalo()
        assert model.embedding_dim == 32
        assert model.n_layers == 3
        assert model.n_heads == 4
        assert model.nn_epochs == 200
        assert model.n_mcmc == 5000

    def test_custom_parameters(self):
        model = BayesianDeepHalo(
            embedding_dim=8, n_layers=2, n_heads=2,
            nn_epochs=10, n_mcmc=100, n_burnin=30,
        )
        assert model.embedding_dim == 8
        assert model.n_mcmc == 100
        assert model.n_burnin == 30

    def test_compute_shares_before_fit(self):
        """RuntimeError before fit."""
        model = BayesianDeepHalo()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.compute_shares(None)

    def test_get_posterior_summary_before_fit(self):
        """RuntimeError before fit."""
        model = BayesianDeepHalo()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_posterior_summary()

    def test_get_sparsity_summary_before_fit(self):
        """RuntimeError before fit."""
        model = BayesianDeepHalo()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_sparsity_summary()

    def test_prepare_data_sets_dims(self):
        """_prepare_data sets T, J, n_features."""
        model = BayesianDeepHalo()
        X = np.random.randn(5, 3, 2).astype(np.float32)
        q = np.abs(np.random.randn(5, 4)).astype(np.float32) + 0.1
        model._prepare_data(X, q)
        assert model.T == 5
        assert model.J == 3
        assert model.n_features == 2
