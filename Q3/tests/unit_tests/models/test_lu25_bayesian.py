"""Unit tests for Lu & Shimizu (2025) Bayesian model components."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from choice_learn.models.lu25_bayesian import (
    Lu25KernelResults,
    Lu25MCMCKernel,
    Lu25Model,
    _ll_per_market,
    _ll_total,
    _shares_all,
    _shares_single,
    rwmh_step,
    sample_gamma,
    sample_phi,
    tmh_step,
)


# Fixtures

@pytest.fixture
def small_data():
    """Small synthetic data for share/likelihood tests."""
    np.random.seed(42)
    T, J, d_X, R0 = 5, 3, 2, 10
    X = tf.constant(np.random.randn(T, J, d_X).astype(np.float32))
    q = tf.constant(np.abs(np.random.randn(T, J + 1)).astype(np.float32))
    v_draws = tf.random.normal((R0, d_X), seed=42)
    rc_indices = tf.constant([0, 1], dtype=tf.int32)
    return X, q, v_draws, rc_indices


@pytest.fixture
def simple_params():
    """Simple parameter values for share/likelihood tests."""
    beta_bar = tf.constant([-1.0, 0.5], dtype=tf.float32)
    r = tf.constant([0.0, 0.0], dtype=tf.float32)
    xi_bar = tf.constant([-1.0] * 5, dtype=tf.float32)
    eta = tf.zeros((5, 3), dtype=tf.float32)
    return beta_bar, r, xi_bar, eta


# --- Spike-and-slab sampling ---

class TestSampleGamma:
    """Tests for sample_gamma."""

    def test_output_shape(self):
        """Output shape matches (T, J)."""
        eta = tf.zeros((5, 3))
        phi = tf.fill((5,), 0.5)
        gamma = sample_gamma(eta, phi, 1e-3, 1.0)
        assert gamma.shape == (5, 3)

    def test_values_binary(self):
        """Output values are 0.0 or 1.0."""
        eta = tf.random.normal((5, 3))
        phi = tf.fill((5,), 0.5)
        gamma = sample_gamma(eta, phi, 1e-3, 1.0)
        g = gamma.numpy()
        assert np.all((g == 0.0) | (g == 1.0))

    def test_large_eta_favors_slab(self):
        """Large |eta| with small tau0_sq should produce gamma=1 (slab)."""
        eta = tf.fill((10, 5), 5.0)
        phi = tf.fill((10,), 0.5)
        gamma = sample_gamma(eta, phi, 1e-6, 10.0)
        assert np.mean(gamma.numpy()) > 0.8


class TestSamplePhi:
    """Tests for sample_phi."""

    def test_output_shape(self):
        """Output shape matches (T,)."""
        gamma = tf.ones((5, 3))
        phi = sample_phi(gamma, 1.0, 1.0, 3)
        assert phi.shape == (5,)

    def test_values_in_unit_interval(self):
        """All values in [0, 1]."""
        gamma = tf.constant(np.random.choice([0.0, 1.0], size=(5, 3)).astype(np.float32))
        phi = sample_phi(gamma, 1.0, 1.0, 3)
        assert np.all(phi.numpy() >= 0.0)
        assert np.all(phi.numpy() <= 1.0)

    def test_all_ones_gamma_high_phi(self):
        """When all gamma=1, phi should be high."""
        gamma = tf.ones((20, 5))
        phi = sample_phi(gamma, 1.0, 1.0, 5)
        assert np.mean(phi.numpy()) > 0.5


# --- MH steps ---

class TestRWMHStep:
    """Tests for rwmh_step."""

    def test_output_shape(self):
        """Output shapes match input."""
        current = tf.constant([1.0, 2.0, 3.0])
        log_fn = lambda x: -0.5 * tf.reduce_sum(x ** 2)
        updated, accepted = rwmh_step(current, log_fn, 1.0)
        assert updated.shape == (3,)

    def test_zero_kappa_stays(self):
        """With kappa=0, proposal equals current so always accepted."""
        current = tf.constant([1.0, 2.0])
        log_fn = lambda x: -0.5 * tf.reduce_sum(x ** 2)
        updated, _ = rwmh_step(current, log_fn, 0.0)
        assert np.allclose(updated.numpy(), current.numpy())

    def test_batched_input(self):
        """Works with batched (T,) input."""
        current = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        log_fn = lambda x: -0.5 * x ** 2  # element-wise
        updated, accepted = rwmh_step(current, log_fn, 0.5)
        assert updated.shape == (5,)
        assert accepted.shape == (5,)


class TestTMHStep:
    """Tests for tmh_step."""

    def test_output_shape(self):
        """Output shape matches input."""
        current = tf.constant([1.0, 2.0])
        log_fn = lambda x: -0.5 * tf.reduce_sum(x ** 2)
        updated, accepted = tmh_step(current, log_fn, 1.0)
        assert updated.shape == (2,)

    def test_finite_output(self):
        """Output is finite with well-behaved quadratic."""
        current = tf.constant([0.5, -0.5])
        log_fn = lambda x: -0.5 * tf.reduce_sum(x ** 2)
        updated, _ = tmh_step(current, log_fn, 1.0)
        assert np.all(np.isfinite(updated.numpy()))


# --- Share computation ---

class TestShareFunctions:
    """Tests for _shares_all and _shares_single."""

    def test_shares_all_shape(self, small_data, simple_params):
        """_shares_all returns (T, J+1)."""
        X, q, v_draws, rc = small_data
        beta, r, xi, eta = simple_params
        shares = _shares_all(beta, r, xi, eta, X, v_draws, rc)
        assert shares.shape == (5, 4)

    def test_shares_single_shape(self, small_data, simple_params):
        """_shares_single returns (J+1,)."""
        X, q, v_draws, rc = small_data
        beta, r, xi, eta = simple_params
        shares = _shares_single(beta, r, xi[0], eta[0], X[0], v_draws, rc)
        assert shares.shape == (4,)

    def test_shares_all_sum_to_one(self, small_data, simple_params):
        """Each market's shares sum to 1."""
        X, q, v_draws, rc = small_data
        beta, r, xi, eta = simple_params
        shares = _shares_all(beta, r, xi, eta, X, v_draws, rc).numpy()
        assert np.allclose(np.sum(shares, axis=1), 1.0, atol=1e-5)

    def test_shares_single_sums_to_one(self, small_data, simple_params):
        """Single market shares sum to 1."""
        X, q, v_draws, rc = small_data
        beta, r, xi, eta = simple_params
        shares = _shares_single(beta, r, xi[0], eta[0], X[0], v_draws, rc).numpy()
        assert np.isclose(np.sum(shares), 1.0, atol=1e-5)

    def test_shares_all_vs_single_agree(self, small_data, simple_params):
        """_shares_all[t] matches _shares_single for each market."""
        X, q, v_draws, rc = small_data
        beta, r, xi, eta = simple_params
        all_shares = _shares_all(beta, r, xi, eta, X, v_draws, rc).numpy()
        for t in range(5):
            single = _shares_single(beta, r, xi[t], eta[t], X[t], v_draws, rc).numpy()
            assert np.allclose(all_shares[t], single, atol=1e-5)

    def test_shares_positive(self, small_data, simple_params):
        """All shares are positive."""
        X, q, v_draws, rc = small_data
        beta, r, xi, eta = simple_params
        shares = _shares_all(beta, r, xi, eta, X, v_draws, rc).numpy()
        assert np.all(shares > 0)


# --- Log-likelihood ---

class TestLogLikelihood:
    """Tests for _ll_per_market and _ll_total."""

    def test_ll_per_market_shape(self, small_data, simple_params):
        """Returns (T,)."""
        X, q, v_draws, rc = small_data
        beta, r, xi, eta = simple_params
        ll = _ll_per_market(beta, r, xi, eta, X, q, v_draws, rc)
        assert ll.shape == (5,)

    def test_ll_total_is_sum(self, small_data, simple_params):
        """_ll_total equals sum of _ll_per_market."""
        X, q, v_draws, rc = small_data
        beta, r, xi, eta = simple_params
        ll_per = _ll_per_market(beta, r, xi, eta, X, q, v_draws, rc)
        ll_tot = _ll_total(beta, r, xi, eta, X, q, v_draws, rc)
        assert np.isclose(float(ll_tot), float(tf.reduce_sum(ll_per)), atol=1e-4)

    def test_ll_negative(self, small_data, simple_params):
        """Log-likelihood should be negative."""
        X, q, v_draws, rc = small_data
        beta, r, xi, eta = simple_params
        ll = _ll_total(beta, r, xi, eta, X, q, v_draws, rc)
        assert float(ll) < 0


# --- Lu25MCMCKernel ---

class TestLu25MCMCKernel:
    """Tests for the MCMC kernel."""

    @pytest.fixture
    def kernel_and_state(self, small_data):
        """Create kernel and initial state for testing."""
        X, q, v_draws, rc = small_data
        T, J, d_X = 5, 3, 2
        kernel = Lu25MCMCKernel(
            X, q, v_draws, rc,
            kappa_beta=0.1, kappa_r=0.01, kappa_xi=0.01, kappa_eta=0.1,
            mu_beta=0.0, V_beta=10.0, mu_xi=0.0, V_xi=10.0, V_r=0.5,
            tau0_sq=1e-3, tau1_sq=1.0, a_phi=1.0, b_phi=1.0,
            use_tmh=False,
        )
        state = (
            tf.zeros(d_X),                # beta
            tf.zeros(d_X),                # r
            tf.zeros(T),                  # xi
            tf.zeros((T, J)),             # eta
            tf.zeros((T, J)),             # gamma
            tf.fill((T,), 0.1),           # phi
        )
        return kernel, state

    def test_is_calibrated(self, kernel_and_state):
        kernel, _ = kernel_and_state
        assert kernel.is_calibrated

    def test_bootstrap_results(self, kernel_and_state):
        kernel, state = kernel_and_state
        kr = kernel.bootstrap_results(state)
        assert isinstance(kr, Lu25KernelResults)
        assert float(kr.acc_beta) == 0.0

    def test_one_step_output_shapes(self, kernel_and_state):
        """State shapes preserved after one sweep."""
        kernel, state = kernel_and_state
        kr = kernel.bootstrap_results(state)
        new_state, new_kr = kernel.one_step(state, kr)

        assert new_state[0].shape == (2,)    # beta
        assert new_state[1].shape == (2,)    # r
        assert new_state[2].shape == (5,)    # xi
        assert new_state[3].shape == (5, 3)  # eta
        assert new_state[4].shape == (5, 3)  # gamma
        assert new_state[5].shape == (5,)    # phi

    def test_one_step_returns_kernel_results(self, kernel_and_state):
        kernel, state = kernel_and_state
        kr = kernel.bootstrap_results(state)
        _, new_kr = kernel.one_step(state, kr)
        assert isinstance(new_kr, Lu25KernelResults)


# --- Lu25Model ---

class TestLu25Model:
    """Tests for Lu25Model initialization and data preparation."""

    def test_default_parameters(self):
        model = Lu25Model()
        assert model.tau0_sq == 1e-3
        assert model.tau1_sq == 1.0
        assert model.a_phi == 1.0
        assert model.n_mcmc == 5000

    def test_custom_parameters(self):
        model = Lu25Model(tau0_sq=0.01, tau1_sq=2.0, n_mcmc=100, use_tmh=False)
        assert model.tau0_sq == 0.01
        assert model.tau1_sq == 2.0
        assert model.n_mcmc == 100
        assert not model.use_tmh

    def test_prepare_data_sets_attributes(self):
        """_prepare_data sets T, J, d_X, rc_indices."""
        model = Lu25Model()
        X = np.random.randn(5, 3, 2).astype(np.float32)
        q = np.abs(np.random.randn(5, 4)).astype(np.float32) + 0.1
        model._prepare_data(X, q)

        assert model.T == 5
        assert model.J == 3
        assert model.d_X == 2
        assert model.d_rc == 2

    def test_init_state_shapes(self):
        """_init_state produces state tuple with correct shapes."""
        model = Lu25Model(use_tmh=False)
        X = np.random.randn(5, 3, 2).astype(np.float32)
        q = np.abs(np.random.randn(5, 4)).astype(np.float32) + 0.1
        model._prepare_data(X, q)
        model._init_state()

        state = model._init_state_tuple
        assert len(state) == 6
        assert state[0].shape == (2,)    # beta
        assert state[1].shape == (2,)    # r
        assert state[2].shape == (5,)    # xi
        assert state[3].shape == (5, 3)  # eta
        assert state[4].shape == (5, 3)  # gamma
        assert state[5].shape == (5,)    # phi

    def test_simple_logit_ols_shapes(self):
        """OLS initialization returns correct shapes."""
        model = Lu25Model()
        X = np.random.randn(5, 3, 2).astype(np.float32)
        q = np.abs(np.random.randn(5, 4)).astype(np.float32) + 0.1
        model._prepare_data(X, q)

        beta, xi = model._simple_logit_ols()
        assert beta.shape == (2,)
        assert xi.shape == (5,)

    def test_simple_logit_ols_finite(self):
        """OLS initialization produces finite values."""
        model = Lu25Model()
        np.random.seed(42)
        X = np.random.randn(5, 3, 2).astype(np.float32)
        q = np.abs(np.random.randn(5, 4)).astype(np.float32) + 0.1
        model._prepare_data(X, q)

        beta, xi = model._simple_logit_ols()
        assert np.all(np.isfinite(beta.numpy()))
        assert np.all(np.isfinite(xi.numpy()))
