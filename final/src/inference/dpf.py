"""DPF + Adam joint MLE wrapper around the Q2 DifferentiableParticleFilter.

Maximises the particle-filter marginal log-likelihood in (mu, phi, sigma, psi,
theta_DH) by Adam, with a selectable ``resampler``: the discrete Q2 resamplers
block the gradient through the resample step, the forward-differentiable ones
(stop-gradient / soft / Sinkhorn OT) do not. The trained latent state is read
from a final forward pass.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from src.inference.q2.filters import DifferentiableParticleFilter as Q2DPF
from src.inference.q2.filters.common import DTYPE
from src.inference.q2.resampling import StopGradientResampler
from src.inference.ssm_wrapper import DeepHaloMacroSSM


def run_dpf_q2_adam(
    choice_model,
    data: dict,
    init: dict | None = None,
    n_particles: int = 200,
    n_particles_state: int | None = None,
    n_steps: int = 100,
    lr: float = 0.05,
    free_psi: bool = True,
    free_dh: bool = True,
    free_mu: bool = True,
    free_phi: bool = True,
    free_sigma: bool = True,
    resampler=None,
    resample_threshold: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Fit the DPF state-space parameters by Adam joint MLE.

    Maximises the particle-filter marginal log-likelihood in (mu, phi, sigma)
    and, when free, psi and the DeepHalo weights, then reads the filtered state
    from a final forward pass at the fitted parameters.

    Parameters
    ----------
    choice_model : object
        DeepHalo choice model exposing ``psi_offer`` and ``deephalo``; its
        trainable weights are co-optimized when ``free_dh`` is set, and its
        ``psi_offer`` is synced with the trained value on exit when
        ``free_psi`` is set.
    data : dict
        Observation/covariate data passed to the SSM wrapper.
    init : dict or None
        Initial values for ``mu``, ``phi``, ``sigma``. Defaults to
        ``{"mu": 0.0, "phi": 0.5, "sigma": 0.5}`` when None.
    n_particles : int
        Number of particles in the differentiable particle filter (training).
    n_particles_state : int or None
        Particles for the final state-recovery forward pass. That forward is
        gradient-free, so it can use more particles than training for a
        lower-variance estimate. None reuses ``n_particles``.
    n_steps : int
        Number of Adam optimization steps.
    lr : float
        Adam learning rate.
    free_psi : bool
        If True, optimize ``psi_offer`` jointly and sync it back into
        ``choice_model`` afterward; otherwise it is held constant.
    free_dh : bool
        If True, co-optimize the DeepHalo trainable weights. When False, the
        observation map ``h`` is precomputed in the SSM wrapper.
    resampler : object or None
        Resampling strategy. When None, defaults to the production
        Scibior stop-gradient resampler: joint MLE needs a
        forward-differentiable resampler, else the resample step blocks the
        gradient and Adam sees a biased estimate.
    resample_threshold : float
        Effective-sample-size fraction below which resampling triggers.
    seed : int
        Seed for the particle-noise random generator.
    verbose : bool
        If True, print progress roughly ten times over the run.

    Returns
    -------
    dict
        Keys ``mu``, ``phi``, ``sigma`` (fitted point estimates); ``x_hat``,
        ``x_std``, ``ess`` (the filtered state from a final forward pass at the
        fitted parameters -- the DPF already filters x_t at every step, so no
        separate bootstrap PF is needed); ``history`` (per-step traces); and
        ``time`` (wall-clock seconds). When ``free_psi`` is set, also includes
        ``psi_offer``.
    """
    if init is None:
        init = {"mu": 0.0, "phi": 0.5, "sigma": 0.5}
    if resampler is None:
        # Joint MLE needs a forward-differentiable resampler: a discrete one
        # blocks the gradient through the resample step and Adam sees a biased
        # estimate. Default to the production Scibior stop-gradient resampler.
        resampler = StopGradientResampler()

    mu_var = tf.Variable(init["mu"], dtype=DTYPE, name="mu")
    phi_raw_var = tf.Variable(np.arctanh(init["phi"]), dtype=DTYPE, name="phi_raw")
    log_sigma_var = tf.Variable(np.log(init["sigma"]), dtype=DTYPE, name="log_sigma")

    # Each SSM param is free by default; setting free_mu/free_phi/free_sigma=False
    # holds it at its init. Fixing the scale/location (sigma, mu) pins the
    # (x, mu, sigma, psi) affine freedom the choice data cannot identify, e.g. set
    # them from an external proxy's AR(1) fit. See context/codebase/identifiability.md.
    free_vars = []
    if free_mu:
        free_vars.append(mu_var)
    if free_phi:
        free_vars.append(phi_raw_var)
    if free_sigma:
        free_vars.append(log_sigma_var)
    if free_psi:
        psi_var = tf.Variable(
            tf.cast(choice_model.psi_offer, DTYPE).numpy(),
            dtype=DTYPE, name="psi_offer",
        )
        free_vars.append(psi_var)
    else:
        psi_var = tf.constant(
            tf.cast(choice_model.psi_offer, DTYPE).numpy(), dtype=DTYPE,
        )

    if free_dh:
        free_vars = free_vars + list(choice_model.deephalo.trainable_weights)

    # Build the SSM *once* in raw-vars mode and reuse the same instance every
    # Adam step. The SSM's parameter properties recompute tanh/exp/softplus
    # from the raw Variables on each access, so gradients still flow back to
    # ``free_vars`` while the @tf.function filter loop only traces once.
    ssm = DeepHaloMacroSSM(
        choice_model, data,
        mu_var=mu_var, phi_raw_var=phi_raw_var, log_sigma_var=log_sigma_var,
        psi_var=psi_var,
        precompute_h=not free_dh,
    )
    ys = ssm.ys_indices

    dpf = Q2DPF(n_particles=n_particles, resampler=resampler,
                resample_threshold=resample_threshold)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Single generator: its internal state advances on every sample, so each
    # Adam step sees fresh particle noise without spawning a new Python
    # ``tf.random.Generator`` object (which would retrace the filter loop).
    rng = tf.random.Generator.from_seed(seed)

    history = {"mu": [], "phi": [], "sigma": [], "neg_log_lik": [], "step": [],
               "psi_offer": []}
    t0 = time.time()

    for step in range(n_steps):
        with tf.GradientTape() as tape:
            res = dpf.filter(ssm, ys, rng=rng)
            log_marg = res.diagnostics["log_marginal_likelihood"]
            neg_log_lik = -log_marg
            loss = neg_log_lik
        grads = tape.gradient(loss, free_vars)
        optimizer.apply_gradients(zip(grads, free_vars))

        mu_val = float(mu_var.numpy())
        phi_val = float(tf.tanh(phi_raw_var).numpy())
        sigma_val = float(tf.exp(log_sigma_var).numpy())
        history["mu"].append(mu_val)
        history["phi"].append(phi_val)
        history["sigma"].append(sigma_val)
        history["neg_log_lik"].append(float(neg_log_lik.numpy()))
        history["step"].append(step)
        if free_psi:
            history["psi_offer"].append(psi_var.numpy().copy())

        if verbose and (step + 1) % max(1, n_steps // 10) == 0:
            print(f"    [{step+1:4d}/{n_steps}]  mu={mu_val:+.3f}  phi={phi_val:+.3f}  "
                  f"sigma={sigma_val:.3f}  -log_lik={float(neg_log_lik.numpy()):.1f}")

    # Final forward pass at the trained parameters to read the filtered state.
    # The DPF filters x_t at every step (m_filt = weighted particle mean), so no
    # separate bootstrap PF is needed (for stop-gradient the forward pass is the
    # standard PF; Scibior et al. 2021 modify only the gradient). This pass is
    # gradient-free, so it can use more particles than training.
    n_state = n_particles if n_particles_state is None else n_particles_state
    state_dpf = Q2DPF(n_particles=n_state, resampler=resampler,
                      resample_threshold=resample_threshold)
    final = state_dpf.filter(ssm, ys, rng=rng)
    m_filt = final.m_filt.numpy()
    P_filt = final.P_filt.numpy()

    out = {
        "mu": float(mu_var.numpy()),
        "phi": float(tf.tanh(phi_raw_var).numpy()),
        "sigma": float(tf.exp(log_sigma_var).numpy()),
        "x_hat": m_filt[:, 0].astype(np.float32),
        "x_std": np.sqrt(P_filt[:, 0, 0]).astype(np.float32),
        "ess": final.diagnostics["ess"].numpy().astype(np.float32),
        "history": history,
        "time": time.time() - t0,
    }
    if free_psi:
        out["psi_offer"] = psi_var.numpy()
        # Sync back into the model so downstream PF calls read the trained psi.
        choice_model.psi_offer.assign(
            tf.cast(psi_var, choice_model.psi_offer.dtype)
        )
    return out
