"""Integration tests for Range-Bearing tracking pipeline."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm import RangeBearing
from src.filters import extended_kalman_filter, unscented_kalman_filter, particle_filter
from src.flows import exact_daum_huang_flow, local_edh_flow
from src.utils.metrics import compute_rmse


@pytest.fixture
def range_bearing_data(rng):
    """Generate range-bearing tracking data."""
    T = 80
    model = RangeBearing(dt=1.0, q=0.1, r_range=0.5, r_bearing=0.05)
    xs, ys = model.simulate(T, rng)

    return {
        'model': model,
        'xs': xs,
        'ys': ys,
        'T': T
    }


class TestEKFOnRangeBearing:
    """Test EKF on range-bearing tracking."""

    def test_ekf_on_range_bearing(self, range_bearing_data):
        """EKF should track range-bearing model."""
        d = range_bearing_data
        model = d['model']

        m_filt, P_filt, cond_nums = extended_kalman_filter(
            model.f, model.h, model.F_jac, model.H_jac,
            model.Q, model.R, model.m0, model.P0, d['ys'],
            angle_indices=[1]
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))

        # Check RMSE is reasonable
        rmse = compute_rmse(m_filt, d['xs'])
        assert rmse < 5.0, f"RMSE too large: {rmse}"


class TestUKFOnRangeBearing:
    """Test UKF on range-bearing tracking."""

    def test_ukf_on_range_bearing(self, range_bearing_data):
        """UKF should track range-bearing model."""
        d = range_bearing_data
        model = d['model']

        m_filt, P_filt, cond_nums = unscented_kalman_filter(
            model.f, model.h, model.Q, model.R, model.m0, model.P0, d['ys'],
            angle_indices=[1]
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))

        # Check RMSE is reasonable
        rmse = compute_rmse(m_filt, d['xs'])
        assert rmse < 5.0, f"RMSE too large: {rmse}"


class TestPFOnRangeBearing:
    """Test Particle Filter on range-bearing tracking."""

    def test_pf_on_range_bearing(self, rng, range_bearing_data):
        """PF should track range-bearing model."""
        d = range_bearing_data
        model = d['model']

        m_filt, P_filt, ess, _ = particle_filter(
            model.f, model.h, model.Q_sampler, model.log_likelihood,
            model.m0, model.P0, d['ys'],
            N_particles=500, rng=rng
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert np.all(ess >= 1)

        # Check RMSE is reasonable
        rmse = compute_rmse(m_filt, d['xs'])
        assert rmse < 10.0, f"RMSE too large: {rmse}"


class TestEDHFlowOnRangeBearing:
    """Test EDH flow on range-bearing model."""

    def test_edh_flow_on_range_bearing(self, rng, range_bearing_data):
        """EDH flow should move particles appropriately."""
        d = range_bearing_data
        model = d['model']

        # Single step test
        N = 200
        particles = rng.multivariate_normal(model.m0, model.P0, size=N)
        y = d['ys'][0]

        # Create batch-compatible f function for flows
        def f_batch(x):
            if x.ndim > 1:
                return (model.F @ x.T).T
            return model.F @ x

        particles_out, log_det_J, _, m_post, P_post = exact_daum_huang_flow(
            particles, model.m0, model.P0,
            f_batch, model.F_jac, model.Q,
            model.h, model.H_jac, model.R, y,
            n_steps=20, redraw=False, rng=rng, filter_type='ekf',
            angle_indices=[1]
        )

        # Check outputs are valid
        assert not np.any(np.isnan(particles_out))
        assert np.all(np.isfinite(log_det_J))


class TestLEDHFlowOnRangeBearing:
    """Test LEDH flow on range-bearing model."""

    def test_ledh_flow_on_range_bearing(self, rng, range_bearing_data):
        """LEDH flow should work on range-bearing model."""
        d = range_bearing_data
        model = d['model']

        # Single step test
        N = 200
        particles = rng.multivariate_normal(model.m0, model.P0, size=N)
        y = d['ys'][0]

        # Create batch-compatible f function for flows
        def f_batch(x):
            if x.ndim > 1:
                return (model.F @ x.T).T
            return model.F @ x

        particles_out, log_det_J, _, m_post, P_post = local_edh_flow(
            particles, model.m0, model.P0,
            f_batch, model.F_jac, model.Q,
            model.h, model.H_jac, model.R, y,
            n_steps=15, store_history=False, redraw=False, rng=rng,
            filter_type='ekf', angle_indices=[1]
        )

        # Check outputs are valid
        assert not np.any(np.isnan(particles_out))
        assert np.all(np.isfinite(log_det_J))


class TestAngleWrapping:
    """Test that angle wrapping is handled correctly."""

    def test_angle_wrapping_handled(self, rng):
        """Filters should handle angle wrapping near +/- pi."""
        # Create model with target that crosses +/- pi boundary
        model = RangeBearing(dt=1.0, q=0.1, r_range=0.5, r_bearing=0.05)
        model.set_initial(
            m0=np.array([-1.0, 0.0, 0.0, 0.5]),  # Start near negative x-axis
            P0=np.diag([0.5, 0.1, 0.5, 0.1])
        )

        T = 50
        xs, ys = model.simulate(T, rng)

        # EKF with angle wrapping
        m_ekf, _, _ = extended_kalman_filter(
            model.f, model.h, model.F_jac, model.H_jac,
            model.Q, model.R, model.m0, model.P0, ys,
            angle_indices=[1]
        )

        # UKF with angle wrapping
        m_ukf, _, _ = unscented_kalman_filter(
            model.f, model.h, model.Q, model.R, model.m0, model.P0, ys,
            angle_indices=[1]
        )

        # Both should produce valid outputs
        assert not np.any(np.isnan(m_ekf))
        assert not np.any(np.isnan(m_ukf))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
