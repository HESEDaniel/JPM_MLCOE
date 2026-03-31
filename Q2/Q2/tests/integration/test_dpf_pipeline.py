"""Integration tests for DPF pipeline on CorenflosLGSSM."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

DTYPE = tf.float64


@pytest.fixture(scope='module')
def corenflos_data():
    """Small CorenflosLGSSM instance with simulated data."""
    from src.ssm import CorenflosLGSSM
    ssm = CorenflosLGSSM(d_x=5, d_y=1, base=0.42, sigma_obs=0.316,
                          dtype=tf.float64)
    rng = np.random.default_rng(42)
    T = 20
    xs, ys = ssm.simulate(T, rng)
    ys_tf = tf.constant(ys, dtype=DTYPE)
    return {'ssm': ssm, 'xs': xs, 'ys_tf': ys_tf, 'T': T}


class TestDPFOnCorenflosLGSSM:
    """DPF end-to-end on the Corenflos(2021) linear Gaussian SSM."""

    def test_dpf_ot_runs(self, corenflos_data):
        """DPF with OT resampling should run on CorenflosLGSSM without error."""
        from src.filters import DifferentiableParticleFilter, FilterResult
        from src.resampling import SinkhornOTResampler

        d = corenflos_data
        dpf = DifferentiableParticleFilter(
            n_particles=50,
            resampler=SinkhornOTResampler(epsilon=0.5, max_iters=30))
        rng_tf = tf.random.Generator.from_seed(7)
        res = dpf.filter(d['ssm'], d['ys_tf'], rng=rng_tf)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (d['T'], 5)
        assert not np.any(np.isnan(res.m_filt.numpy()))

    def test_dpf_soft_runs(self, corenflos_data):
        """DPF with soft resampling should run on CorenflosLGSSM."""
        from src.filters import DifferentiableParticleFilter
        from src.resampling import SoftResampler

        d = corenflos_data
        dpf = DifferentiableParticleFilter(
            n_particles=50, resampler=SoftResampler(alpha=0.5))
        rng_tf = tf.random.Generator.from_seed(7)
        res = dpf.filter(d['ssm'], d['ys_tf'], rng=rng_tf)

        assert res.m_filt.shape == (d['T'], 5)
        assert not np.any(np.isnan(res.m_filt.numpy()))

    def test_dpf_tracks_signal(self, corenflos_data):
        """DPF estimate should be closer to truth than zero baseline."""
        from src.filters import DifferentiableParticleFilter

        d = corenflos_data
        dpf = DifferentiableParticleFilter(n_particles=100)
        rng_tf = tf.random.Generator.from_seed(7)
        res = dpf.filter(d['ssm'], d['ys_tf'], rng=rng_tf)

        mse_filter = np.mean((res.m_filt.numpy() - d['xs']) ** 2)
        mse_prior = np.mean(d['xs'] ** 2)
        assert mse_filter < mse_prior


class TestELBOPipeline:
    """ELBO computation end-to-end."""

    def test_elbo_finite(self, corenflos_data):
        """dpf_elbo should return finite ELBO and positive mean ESS."""
        from src.utils.elbo import dpf_elbo, log_gaussian_full
        from src.filters.base_particle import BaseParticleFilter
        from src.resampling import SinkhornOTResampler

        d = corenflos_data
        ssm = d['ssm']
        N = 30
        rng_tf = tf.random.Generator.from_seed(11)
        particles = BaseParticleFilter.init_particles(ssm.m0, ssm.P0, N, rng_tf)

        # Simple bootstrap proposal: propagate + dynamics noise
        def proposal_fn(particles, y_t, rng):
            noise = rng.normal(shape=tf.shape(particles), dtype=DTYPE)
            x_new = ssm.f_batch(particles) + noise
            # log q = log N(x_new; Ax, Q=I) = transition density
            log_q = log_gaussian_full(
                x_new, ssm.f_batch(particles), ssm.Q_inv, ssm.log_det_Q)
            return x_new, log_q

        resampler = SinkhornOTResampler(epsilon=0.5, max_iters=30)
        elbo, mean_ess = dpf_elbo(
            particles, d['ys_tf'], ssm, proposal_fn, resampler, rng_tf)

        assert np.isfinite(elbo.numpy()), f"ELBO not finite: {elbo.numpy()}"
        assert mean_ess.numpy() > 0.0, f"Mean ESS not positive: {mean_ess.numpy()}"


class TestStochasticFlowOnLinearSSM:
    """StochasticFlow integration test on standard LinearGaussianSSM."""

    def test_stochastic_flow_tracks_signal(self):
        from src.ssm import LinearGaussianSSM
        from src.flows import StochasticFlow

        A = np.array([[0.9, 0.1], [0.0, 0.95]])
        B = np.sqrt(0.1) * np.eye(2)
        C = np.eye(2)
        D = np.sqrt(0.1) * np.eye(2)
        Sigma = np.eye(2)
        ssm = LinearGaussianSSM(A, B, C, D, Sigma)

        rng = np.random.default_rng(42)
        T = 30
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        Q_diff = tf.eye(2, dtype=DTYPE) * 0.3
        sf = StochasticFlow(n_particles=100, n_flow_steps=10, Q_diff=Q_diff)
        res = sf.filter(ssm, ys_tf, rng=tf.random.Generator.from_seed(42))

        mse_filter = np.mean((res.m_filt.numpy() - xs) ** 2)
        mse_prior = np.mean(xs ** 2)
        assert mse_filter < mse_prior
        assert not np.any(np.isnan(res.m_filt.numpy()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
