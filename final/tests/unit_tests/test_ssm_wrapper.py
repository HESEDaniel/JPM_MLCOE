"""Unit tests for the AR(1) + categorical state-space wrapper (DeepHaloMacroSSM).

Covers the parameterization and the stationary-law initialization the report
relies on: phi = tanh(phi_raw) stays in (-1, 1), sigma = exp(log_sigma) stays
positive, and (m0, P0, Q) match the stationary AR(1) moments.
"""

import numpy as np
import tensorflow as tf

from src.datasets.dgp_macro import DGPConfig, simulate_macro_choice_dgp
from src.choice_models.macro_deephalo import MacroDeepHalo
from src.inference.ssm_wrapper import DeepHaloMacroSSM
from src.inference.q2.filters.common import DTYPE


def _ssm(mu=0.3, phi=0.9, sigma=0.5):
    """Build a DeepHaloMacroSSM in raw-variable mode with known (mu, phi, sigma)."""
    data = simulate_macro_choice_dgp(
        scenario="B", cfg=DGPConfig(M=8, K_min=2, K_max=4, T=20, N_t=5), seed=0)
    model = MacroDeepHalo(M=data["M"])
    return DeepHaloMacroSSM(
        model, data,
        mu_var=tf.Variable(mu, dtype=DTYPE),
        phi_raw_var=tf.Variable(np.arctanh(phi), dtype=DTYPE),
        log_sigma_var=tf.Variable(np.log(sigma), dtype=DTYPE),
        psi_var=tf.Variable(model.psi_offer.numpy(), dtype=DTYPE),
    )


def test_reparameterization_constraints():
    """phi = tanh(phi_raw) stays in (-1, 1) and sigma = exp(log_sigma) stays positive."""
    ssm = _ssm(phi=0.9, sigma=0.5)
    assert -1.0 < float(ssm.phi) < 1.0
    assert abs(float(ssm.phi) - 0.9) < 1e-5
    assert float(ssm.sigma) > 0.0
    assert abs(float(ssm.sigma) - 0.5) < 1e-5


def test_stationary_initialization():
    """m0, P0, Q match the stationary AR(1) law mu/(1-phi), sigma^2/(1-phi^2), sigma^2."""
    mu, phi, sigma = 0.3, 0.9, 0.5
    ssm = _ssm(mu, phi, sigma)
    assert abs(float(ssm.m0[0]) - mu / (1 - phi)) < 1e-4
    assert abs(float(ssm.P0[0, 0]) - sigma ** 2 / (1 - phi ** 2)) < 1e-4
    assert abs(float(ssm.Q[0, 0]) - sigma ** 2) < 1e-5


def test_ar1_transition():
    """f_batch propagates the deterministic mean mu + phi * x."""
    mu, phi = 0.3, 0.9
    ssm = _ssm(mu, phi)
    x = tf.constant([[1.0], [-2.0]], dtype=DTYPE)
    fx = ssm.f_batch(x).numpy()
    assert np.allclose(fx[:, 0], [mu + phi * 1.0, mu + phi * (-2.0)], atol=1e-4)
