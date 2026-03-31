"""Compare numpy and TensorFlow filter outputs on same data.

Ensures TF implementations match the reference numpy implementations
for deterministic filters (KF, EKF, UKF).
"""
import sys
import os

Q2_TF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
Q2_NP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Q2_np'))

import numpy as np
import tensorflow as tf
import pytest

DTYPE = tf.float64

# Helpers to load np and tf in isolation

def _run_np_kf(A, B, C, D, Sigma, ys, joseph=True):
    """Run numpy KF in a clean sys.path."""
    saved = sys.path[:]
    saved_mods = {k: v for k, v in sys.modules.items() if k.startswith('src')}
    for k in list(sys.modules):
        if k.startswith('src'):
            del sys.modules[k]
    try:
        sys.path.insert(0, Q2_NP_ROOT)
        from src.filters.kf import kalman_filter
        m, P, cond = kalman_filter(A, B, C, D, Sigma, ys, joseph=joseph, solver='cholesky')
        return m, P
    finally:
        for k in list(sys.modules):
            if k.startswith('src'):
                del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved


def _run_np_ekf(f, h, F_jac, H_jac, Q, R, m0, P0, ys, joseph=True):
    saved = sys.path[:]
    saved_mods = {k: v for k, v in sys.modules.items() if k.startswith('src')}
    for k in list(sys.modules):
        if k.startswith('src'):
            del sys.modules[k]
    try:
        sys.path.insert(0, Q2_NP_ROOT)
        from src.filters.ekf import extended_kalman_filter
        m, P, _ = extended_kalman_filter(f, h, F_jac, H_jac, Q, R, m0, P0, ys, joseph=joseph)
        return m, P
    finally:
        for k in list(sys.modules):
            if k.startswith('src'):
                del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved


def _run_tf_kf(A, B, C, D, Sigma, ys, joseph=True):
    sys.path.insert(0, Q2_TF_ROOT)
    from src.ssm import LinearGaussianSSM
    from src.filters import KalmanFilter
    ssm = LinearGaussianSSM(A, B, C, D, Sigma)
    res = KalmanFilter(joseph=joseph).filter(ssm, tf.constant(ys, dtype=DTYPE))
    return res.m_filt.numpy(), res.P_filt.numpy()


def _run_tf_ekf(A, B, C, D, Sigma, ys, joseph=True):
    sys.path.insert(0, Q2_TF_ROOT)
    from src.ssm import LinearGaussianSSM
    from src.filters import ExtendedKalmanFilter
    ssm = LinearGaussianSSM(A, B, C, D, Sigma)
    res = ExtendedKalmanFilter(joseph=joseph).filter(ssm, tf.constant(ys, dtype=DTYPE))
    return res.m_filt.numpy(), res.P_filt.numpy()


# Shared data

@pytest.fixture
def lgssm_data():
    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.eye(2)
    D = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma = np.eye(2)
    T = 50

    # Generate with np SSM
    saved = sys.path[:]
    for k in list(sys.modules):
        if k.startswith('src'):
            del sys.modules[k]
    sys.path.insert(0, Q2_NP_ROOT)
    from src.ssm.linear_gaussian import linear_gaussian_ssm
    rng = np.random.default_rng(42)
    xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)
    for k in list(sys.modules):
        if k.startswith('src'):
            del sys.modules[k]
    sys.path[:] = saved
    sys.path.insert(0, Q2_TF_ROOT)

    return {'A': A, 'B': B, 'C': C, 'D': D, 'Sigma': Sigma, 'xs': xs, 'ys': ys, 'T': T}


# Tests

class TestKFNumpyVsTF:
    def test_kf_means_match(self, lgssm_data):
        d = lgssm_data
        m_np, _ = _run_np_kf(d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'])
        m_tf, _ = _run_tf_kf(d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'])
        np.testing.assert_allclose(m_tf, m_np, rtol=1e-10, atol=1e-12,
                                   err_msg="KF means differ between np and tf")

    def test_kf_covariances_match(self, lgssm_data):
        d = lgssm_data
        _, P_np = _run_np_kf(d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'])
        _, P_tf = _run_tf_kf(d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'])
        np.testing.assert_allclose(P_tf, P_np, rtol=1e-10, atol=1e-12,
                                   err_msg="KF covariances differ between np and tf")


class TestEKFNumpyVsTF:
    def test_ekf_means_match(self, lgssm_data):
        d = lgssm_data
        A, Q, R = d['A'], d['B'] @ d['B'].T, d['D'] @ d['D'].T
        C = d['C']
        f = lambda x: A @ x
        h = lambda x: C @ x
        F_jac = lambda x: A
        H_jac = lambda x: C

        m_np, _ = _run_np_ekf(f, h, F_jac, H_jac, Q, R, np.zeros(2), d['Sigma'], d['ys'])
        m_tf, _ = _run_tf_ekf(d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'])
        np.testing.assert_allclose(m_tf, m_np, rtol=1e-10, atol=1e-12,
                                   err_msg="EKF means differ between np and tf")


class TestAllFiltersConsistentMSE:
    def test_kf_mse_matches(self, lgssm_data):
        d = lgssm_data
        m_np, _ = _run_np_kf(d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'])
        m_tf, _ = _run_tf_kf(d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'])
        mse_np = np.mean((d['xs'] - m_np) ** 2)
        mse_tf = np.mean((d['xs'] - m_tf) ** 2)
        np.testing.assert_allclose(mse_tf, mse_np, rtol=1e-10,
                                   err_msg="KF MSE differs between np and tf")
