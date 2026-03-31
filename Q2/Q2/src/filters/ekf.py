"""Extended Kalman Filter (TensorFlow)."""
import tensorflow as tf

from .base import BaseFilter, FilterResult
from .common import joseph_update, standard_update, wrap_angles,\
    compute_kalman_gain, cond_number, DTYPE


class ExtendedKalmanFilter(BaseFilter):
    """Extended Kalman Filter for nonlinear SSM."""

    def __init__(self, joseph: bool = True, angle_indices=None):
        self.joseph = joseph
        self.angle_indices = angle_indices

    def predict(self, m, P, ssm, **kwargs):
        F = ssm.F_jac(m)
        m_pred = ssm.f(m)
        P_pred = F @ P @ tf.transpose(F) + ssm.Q
        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, ssm, **kwargs):
        H = ssm.H_jac(m_pred)
        K, S = compute_kalman_gain(P_pred, H, ssm.R)
        innov = wrap_angles(y - ssm.h(m_pred), self.angle_indices)
        m_post = m_pred + tf.linalg.matvec(K, innov)
        P_post = joseph_update(P_pred, K, H, ssm.R) if self.joseph \
            else standard_update(P_pred, K, H)
        return m_post, P_post

    def filter(self, ssm, ys, **kwargs):
        m_filt, P_filt, cond_nums = self._filter_loop(ssm, ys)
        return FilterResult(m_filt, P_filt, {'cond_nums': cond_nums})

    @tf.function
    def _filter_loop(self, ssm, ys):
        T = tf.shape(ys)[0]
        m, P = ssm.m0, ssm.P0
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
