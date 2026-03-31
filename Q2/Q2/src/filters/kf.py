"""Kalman Filter (TensorFlow)."""
import tensorflow as tf

from .base import BaseFilter, FilterResult
from .common import joseph_update, standard_update, cond_number, DTYPE


class KalmanFilter(BaseFilter):
    """Kalman Filter for Linear Gaussian SSM.

    Requires a LinearGaussianSSM with attributes A, C.
    """

    def __init__(self, joseph: bool = True):
        self.joseph = joseph

    def predict(self, m, P, ssm, **kwargs):
        A, Q = ssm.A, ssm.Q
        m_pred = tf.linalg.matvec(A, m)
        P_pred = A @ P @ tf.transpose(A) + Q
        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, ssm, **kwargs):
        C, R = ssm.C, ssm.R
        S = C @ P_pred @ tf.transpose(C) + R
        L = tf.linalg.cholesky(S)
        K = tf.transpose(tf.linalg.cholesky_solve(L, C @ P_pred))
        innov = y - tf.linalg.matvec(C, m_pred)
        m_post = m_pred + tf.linalg.matvec(K, innov)
        P_post = joseph_update(P_pred, K, C, R) if self.joseph else standard_update(P_pred, K, C)
        return m_post, P_post

    def filter(self, ssm, ys, **kwargs):
        m_filt, P_filt, cond_nums = self._filter_loop(ssm, ys)
        return FilterResult(m_filt, P_filt, {'cond_nums': cond_nums})

    @tf.function
    def _filter_loop(self, ssm, ys):
        T = tf.shape(ys)[0]
        m, P = tf.zeros(ssm.state_dim, dtype=DTYPE), ssm.P0
        m_arr = tf.TensorArray(DTYPE, size=T)
        P_arr = tf.TensorArray(DTYPE, size=T)
        c_arr = tf.TensorArray(DTYPE, size=T)
        for t in tf.range(T):
            m_pred, P_pred = self.predict(m, P, ssm)
            m, P = self.update(m_pred, P_pred, ys[t], ssm)
            m_arr = m_arr.write(t, m)
            P_arr = P_arr.write(t, P)
            c_arr = c_arr.write(t, cond_number(P))
        return m_arr.stack(), P_arr.stack(), c_arr.stack()
