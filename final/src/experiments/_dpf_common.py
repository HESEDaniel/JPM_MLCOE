"""Shared resampler and psi-prior helpers for the Experiment 4 DPF script."""
from __future__ import annotations

import numpy as np

from src.inference.q2.resampling import (
    NoResampling, SoftResampler, StopGradientResampler, SinkhornOTResampler,
)

PSI_PRIOR_MAGNITUDE = 0.5

RESAMPLER_NAMES = [
    "no_resample", "soft", "stop_gradient",
    "sinkhorn_eps0.5", "sinkhorn_eps1.0", "sinkhorn_eps2.0",
]


def build_resampler(name: str):
    """Map a resampler name to its Q2 resampler instance."""
    if name == "no_resample":   return NoResampling()
    if name == "soft":          return SoftResampler(alpha=0.5)
    if name == "stop_gradient": return StopGradientResampler()
    if name == "sinkhorn_eps0.5":  return SinkhornOTResampler(epsilon=0.5, scaling=0.9, max_iters=100)
    if name == "sinkhorn_eps1.0":  return SinkhornOTResampler(epsilon=1.0, scaling=0.9, max_iters=100)
    if name == "sinkhorn_eps2.0":  return SinkhornOTResampler(epsilon=2.0, scaling=0.9, max_iters=100)
    raise ValueError(f"unknown resampler: {name!r}")


def psi_prior_from_true_sign(psi_true: np.ndarray, magnitude: float = PSI_PRIOR_MAGNITUDE) -> np.ndarray:
    """Build a synthetic psi prior as sign(psi_true) * magnitude.

    The synthetic analog of an economist's prior on each offer's cyclical
    character: a domain expert classifies offers as pro-cyclical (+) or
    counter-cyclical (-), and magnitude is a subjective scale prior. Only the
    signs of psi_true are used.

    Parameters
    ----------
    psi_true : np.ndarray
        True per-offer psi values, used only for their signs.
    magnitude : float
        Scale applied to each sign bit.

    Returns
    -------
    np.ndarray
        Float32 prior of the same shape as psi_true.
    """
    sign = np.sign(psi_true).astype(np.float32)
    sign[sign == 0.0] = 1.0   # break ties deterministically
    return (sign * magnitude).astype(np.float32)
