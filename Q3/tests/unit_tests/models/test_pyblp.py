"""Unit tests for PyBLP estimator."""

import numpy as np
import pandas as pd
import pytest

from choice_learn.models.pyblp import PyBLPEstimator


# Fixtures

@pytest.fixture
def small_data():
    """Small synthetic data for product data building tests."""
    np.random.seed(42)
    T, J = 5, 3
    X = np.random.randn(T, J, 2).astype(np.float32)
    q = np.abs(np.random.randn(T, J + 1)).astype(np.float32)
    cost_shock = np.random.randn(T, J).astype(np.float32)
    return X, q, cost_shock, T, J


@pytest.fixture
def estimator_cost_iv():
    return PyBLPEstimator(iv_type="cost_iv", n_draws=100, seed=42)


@pytest.fixture
def estimator_weak_iv():
    return PyBLPEstimator(iv_type="weak_iv", n_draws=100, seed=42)


class TestPyBLPEstimatorInit:
    """Instantiation tests."""

    def test_default_parameters(self):
        est = PyBLPEstimator()
        assert est.iv_type == "cost_iv"
        assert est.n_draws == 1000
        assert est.sigma_init == 1.0
        assert est.seed == 42

    def test_custom_parameters(self):
        est = PyBLPEstimator(iv_type="weak_iv", n_draws=500, sigma_init=2.0, seed=0)
        assert est.iv_type == "weak_iv"
        assert est.n_draws == 500
        assert est.sigma_init == 2.0
        assert est.seed == 0

    def test_not_fitted_initially(self):
        est = PyBLPEstimator()
        assert not est.is_fitted


class TestBuildProductData:
    """Tests for _build_product_data."""

    def test_returns_dataframe(self, estimator_cost_iv, small_data):
        X, q, cost_shock, T, J = small_data
        df = estimator_cost_iv._build_product_data(X, q, cost_shock)
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_length(self, estimator_cost_iv, small_data):
        """DataFrame has T*J rows."""
        X, q, cost_shock, T, J = small_data
        df = estimator_cost_iv._build_product_data(X, q, cost_shock)
        assert len(df) == T * J

    def test_cost_iv_columns(self, estimator_cost_iv, small_data):
        """Cost IV produces correct instrument columns."""
        X, q, cost_shock, T, J = small_data
        df = estimator_cost_iv._build_product_data(X, q, cost_shock)
        required = ['market_ids', 'shares', 'prices', 'w',
                     'demand_instruments0', 'demand_instruments1', 'demand_instruments2']
        for col in required:
            assert col in df.columns

    def test_weak_iv_columns(self, estimator_weak_iv, small_data):
        """Weak IV instruments are w^2, w^3, w^4."""
        X, q, cost_shock, T, J = small_data
        df = estimator_weak_iv._build_product_data(X, q, None)
        w = df['w'].values
        assert np.allclose(df['demand_instruments0'].values, w ** 2)
        assert np.allclose(df['demand_instruments1'].values, w ** 3)
        assert np.allclose(df['demand_instruments2'].values, w ** 4)

    def test_shares_positive(self, estimator_cost_iv, small_data):
        """Inside shares are positive."""
        X, q, cost_shock, T, J = small_data
        df = estimator_cost_iv._build_product_data(X, q, cost_shock)
        assert np.all(df['shares'].values > 0)

    def test_market_ids_correct(self, estimator_cost_iv, small_data):
        """market_ids repeat 0..T-1, each J times."""
        X, q, cost_shock, T, J = small_data
        df = estimator_cost_iv._build_product_data(X, q, cost_shock)
        expected = np.repeat(np.arange(T), J)
        assert np.array_equal(df['market_ids'].values, expected)


class TestPyBLPEstimatorErrors:
    """Error handling tests."""

    def test_fit_cost_iv_without_cost_shock(self, small_data):
        """cost_iv requires cost_shock."""
        X, q, _, T, J = small_data
        est = PyBLPEstimator(iv_type="cost_iv")
        with pytest.raises(ValueError, match="cost_shock required"):
            est.fit(X, q, cost_shock=None)

    def test_get_results_before_fit(self):
        """get_results raises RuntimeError before fit."""
        est = PyBLPEstimator()
        with pytest.raises(RuntimeError, match="Call fit"):
            est.get_results()

    def test_unknown_iv_type(self, small_data):
        """Unknown iv_type raises ValueError."""
        X, q, cost_shock, T, J = small_data
        est = PyBLPEstimator(iv_type="bad_iv")
        with pytest.raises(ValueError, match="Unknown iv_type"):
            est._build_product_data(X, q, cost_shock)
