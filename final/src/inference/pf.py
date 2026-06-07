"""Production-friendly wrapper around the Q2 verbatim bootstrap PF.

Thin wrapper that adapts ``src.inference.q2.filters.ParticleFilter``
(the Q2 paper code + cumulative-weight fix + MLE patch) to the
``(choice_model, data, mu, phi, sigma) -> dict`` interface that the rest of
this codebase expects.

The algorithm itself --- SISR, cumulative log-weight update, ESS-triggered
systematic resampling --- lives in ``q2/filters/pf.py``; this file only
constructs ``DeepHaloMacroSSM`` from the trained choice model and the DGP
output, runs the Q2 PF, and reshapes the ``FilterResult`` into the dict that
experiments and validation scripts consume.

There is a single source of truth for the PF algorithm at
``q2/filters/pf.py``; this file is just the gateway.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.inference.q2.filters import ParticleFilter as Q2ParticleFilter
from src.inference.ssm_wrapper import DeepHaloMacroSSM


def run_pf_q2(
    choice_model,
    data: dict,
    mu: float,
    phi: float,
    sigma: float,
    n_particles: int = 500,
    resample_threshold: float = 0.5,
    seed: int = 42,
    export_particles: bool = False,
) -> dict:
    """Bootstrap PF on the dynamic Deep-Halo SSM, returning a moments dict.

    Constructs ``DeepHaloMacroSSM`` from the trained choice model and the DGP
    output, runs the verbatim Q2 bootstrap PF, and reshapes the
    ``FilterResult`` into the dict that experiments and validation scripts
    consume.

    Parameters
    ----------
    choice_model
        Trained Deep-Halo choice model used to build the SSM emission.
    data : dict
        DGP output passed through to ``DeepHaloMacroSSM``.
    mu : float
        AR(1) latent-state mean parameter.
    phi : float
        AR(1) latent-state persistence parameter.
    sigma : float
        AR(1) latent-state innovation standard deviation.
    n_particles : int
        Number of particles used by the Q2 PF.
    resample_threshold : float
        ESS fraction below which systematic resampling is triggered.
    seed : int
        Seed for the TensorFlow random generator driving the PF.
    export_particles : bool
        If True, also return the per-step particles and weights.

    Returns
    -------
    dict with keys
        x_hat (T,)           posterior mean trajectory
        x_std (T,)           posterior standard-deviation trajectory
        ess (T,)             pre-resample ESS per step
        log_marginal_lik     scalar log p(y_{1:T})
        resample_count       int, total resampling events
        particles (T, N)     present only when export_particles is True
        weights (T, N)       present only when export_particles is True
    """
    ssm = DeepHaloMacroSSM(
        choice_model, data,
        mu=mu, phi=phi, sigma=sigma,
        precompute_h=True,
    )
    pf = Q2ParticleFilter(n_particles=n_particles,
                          resample_threshold=resample_threshold,
                          export_particles=export_particles)
    rng = tf.random.Generator.from_seed(seed)
    result = pf.filter(ssm, ssm.ys_indices, rng=rng)

    # ``FilterResult.m_filt`` is (T, state_dim) = (T, 1); P_filt is (T, 1, 1).
    m_filt = result.m_filt.numpy()
    P_filt = result.P_filt.numpy()
    diag = result.diagnostics
    out = {
        "x_hat": m_filt[:, 0].astype(np.float32),
        "x_std": np.sqrt(P_filt[:, 0, 0]).astype(np.float32),
        "ess": diag["ess"].numpy().astype(np.float32),
        "log_marginal_lik": float(diag["log_marginal_lik"].numpy()),
        "resample_count": int(diag["resample_count"].numpy()),
    }
    if export_particles:
        # particles: (T, N, n_x=1) -> (T, N); weights: (T, N)
        out["particles"] = diag["particles"].numpy()[:, :, 0].astype(np.float32)
        out["weights"] = diag["weights"].numpy().astype(np.float32)
    return out
