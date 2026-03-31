"""Numerical equivalence tests: Q2 TensorFlow vs Q2_np NumPy implementations.

Compares exact matrix formulas and structural properties between np and TF
implementations to verify that the TF refactoring preserves correctness.

Deterministic computations (matrices) are compared bit-exactly.
Stochastic methods (particle filters) are validated by TF-only tests
(test_lgssm_pipeline.py, test_sv_pipeline.py) since RNG backends differ.
"""
import sys
import os

Q2_TF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
Q2_NP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Q2_np'))

import numpy as np
import tensorflow as tf
import pytest

DTYPE = tf.float64

# --Data generation (deterministic, shared by all tests) -------------

def _make_linear_data(seed=42, T=30):
    """Generate 2D linear Gaussian data from numpy reference."""
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    Q = 0.1 * np.eye(2)
    H = np.eye(2)
    R = 0.1 * np.eye(2)
    m0 = np.zeros(2)
    P0 = np.eye(2)

    rng = np.random.default_rng(seed)
    xs = np.zeros((T, 2))
    ys = np.zeros((T, 2))
    x = rng.multivariate_normal(m0, P0)
    for t in range(T):
        x = A @ x + rng.multivariate_normal(np.zeros(2), Q)
        y = H @ x + rng.multivariate_normal(np.zeros(2), R)
        xs[t], ys[t] = x, y

    return {
        'A': A, 'Q': Q, 'H': H, 'R': R, 'm0': m0, 'P0': P0,
        'xs': xs, 'ys': ys, 'T': T,
    }


@pytest.fixture(scope='module')
def linear_data():
    return _make_linear_data()


# --Numpy runner helpers ---------------------------------------------

def _with_np_path(fn):
    """Run fn() with Q2_np on sys.path, isolating src modules."""
    saved = sys.path[:]
    saved_mods = {k: v for k, v in sys.modules.items() if k.startswith('src')}
    for k in list(sys.modules):
        if k.startswith('src'):
            del sys.modules[k]
    try:
        sys.path.insert(0, Q2_NP_ROOT)
        return fn()
    finally:
        for k in list(sys.modules):
            if k.startswith('src'):
                del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved


def _with_tf_path(fn):
    """Run fn() with Q2 TF on sys.path."""
    sys.path.insert(0, Q2_TF_ROOT)
    return fn()


# --Test: EDH matrix computation (exact match) ----------------------

class TestEDHMatricesExact:
    """EDH matrix formulas should match exactly between np and TF."""

    def test_edh_matrices_match(self, linear_data):
        d = linear_data
        m = d['m0']
        P = d['P0']
        H = d['H']
        R = d['R']
        y = d['ys'][0]
        lam = 0.5

        def run_np():
            from src.flows.edh import compute_edh_matrices
            A_np, b_np = compute_edh_matrices(m, P, H, R, y, lam, m, lambda x: H @ x)
            return A_np, b_np

        A_np, b_np = _with_np_path(run_np)

        def run_tf():
            from src.flows.edh import compute_edh_matrices
            A_tf, b_tf = compute_edh_matrices(
                tf.constant(m, dtype=DTYPE), tf.constant(P, dtype=DTYPE),
                tf.constant(H, dtype=DTYPE), tf.constant(R, dtype=DTYPE),
                tf.constant(y, dtype=DTYPE), tf.constant(lam, dtype=DTYPE),
                tf.constant(m, dtype=DTYPE),
                lambda x: tf.linalg.matvec(tf.constant(H, dtype=DTYPE), x))
            return A_tf.numpy(), b_tf.numpy()

        A_tf, b_tf = _with_tf_path(run_tf)

        np.testing.assert_allclose(A_tf, A_np, rtol=1e-10, atol=1e-12,
                                   err_msg="EDH A matrices differ")
        np.testing.assert_allclose(b_tf, b_np, rtol=1e-10, atol=1e-12,
                                   err_msg="EDH b vectors differ")


# --Test: LEDH matrix computation (exact match) ---------------------

class TestLEDHMatricesExact:
    """LEDH matrix formulas should match exactly between np and TF."""

    def test_ledh_matrices_match(self, linear_data):
        d = linear_data
        x_i = np.array([1.5, 0.5])
        m = d['m0']
        P = d['P0']
        R = d['R']
        H = d['H']
        y = d['ys'][0]
        lam = 0.5

        def run_np():
            from src.flows.ledh import compute_ledh_matrices
            R_inv = np.linalg.inv(R)
            A_np, b_np = compute_ledh_matrices(
                x_i, m, P, lambda x: H @ x, lambda x: H, R, y, lam, R_inv)
            return A_np, b_np

        A_np, b_np = _with_np_path(run_np)

        def run_tf():
            from src.flows.ledh import compute_ledh_matrices
            R_inv = tf.linalg.inv(tf.constant(R, dtype=DTYPE))
            A_tf, b_tf = compute_ledh_matrices(
                tf.constant(x_i, dtype=DTYPE),
                tf.constant(m, dtype=DTYPE), tf.constant(P, dtype=DTYPE),
                lambda x: tf.linalg.matvec(tf.constant(H, dtype=DTYPE), x),
                lambda x: tf.constant(H, dtype=DTYPE),
                tf.constant(R, dtype=DTYPE), tf.constant(y, dtype=DTYPE),
                tf.constant(lam, dtype=DTYPE), R_inv)
            return A_tf.numpy(), b_tf.numpy()

        A_tf, b_tf = _with_tf_path(run_tf)

        np.testing.assert_allclose(A_tf, A_np, rtol=1e-10, atol=1e-12,
                                   err_msg="LEDH A matrices differ")
        np.testing.assert_allclose(b_tf, b_np, rtol=1e-10, atol=1e-12,
                                   err_msg="LEDH b vectors differ")


# --Test: Localization matrix (exact match) --------------------------

class TestLocalizationExact:
    """Localization matrix should match exactly."""

    def test_localization_matrix_match(self):
        def run_np():
            from src.flows.rkhs_pff import localization_matrix
            return localization_matrix(8, r_in=3.0)

        C_np = _with_np_path(run_np)

        def run_tf():
            from src.flows.rkhs_pff import localization_matrix
            return localization_matrix(8, r_in=3.0).numpy()

        C_tf = _with_tf_path(run_tf)

        np.testing.assert_allclose(C_tf, C_np, rtol=1e-12,
                                   err_msg="Localization matrices differ")


# --Test: Flow output contract ----------------------------------------

class TestFlowOutputContract:
    """Verify flows return FilterResult with expected diagnostics."""

    def test_all_flows_return_filter_result(self, linear_data):
        d = linear_data
        sys.path.insert(0, Q2_TF_ROOT)
        from src.ssm import LinearGaussianSSM
        from src.flows import EDHFlow, LEDHFlow
        from src.filters.base import FilterResult
        B = np.linalg.cholesky(d['Q'])
        D_mat = np.linalg.cholesky(d['R'])
        ssm = LinearGaussianSSM(d['A'], B, d['H'], D_mat, d['P0'])
        ys_tf = tf.constant(d['ys'], dtype=DTYPE)

        for FlowCls in [EDHFlow, LEDHFlow]:
            flow = FlowCls(n_particles=20, n_flow_steps=3)
            res = flow.filter(ssm, ys_tf, rng=tf.random.Generator.from_seed(42))
            assert isinstance(res, FilterResult), f"{FlowCls.__name__} should return FilterResult"
            assert 'ess' in res.diagnostics, f"{FlowCls.__name__} should have ESS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
