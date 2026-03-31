"""ELBO estimation via particle filters.

The Evidence Lower BOund (ELBO) for a state-space model with parameters theta
is estimated by running a particle filter and accumulating incremental
log-likelihood estimates:

    ELBO = sum_t log(1/N * sum_i w_t^i)

where w_t^i are unnormalized importance weights at time t.

For differentiable PFs, the ELBO gradient w.r.t. model/proposal parameters
flows through the weight computation and (if using OT resampling) through
the resampled particle positions.

References:
    - Corenflos et al. (2021), Section 5
    - Chen et al. (2023), Section 6.3
"""
import numpy as np
import tensorflow as tf

DTYPE = tf.float64

_LOG_2PI = tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))


def log_gaussian_diag(x, mean, log_var):
    """Log N(x; mean, diag(exp(log_var))).

    Parameters
    ----------
    x : tf.Tensor [N, d]
    mean : tf.Tensor [N, d]
    log_var : tf.Tensor [d]
        Log of diagonal variance entries.

    Returns
    -------
    tf.Tensor [N]
    """
    d = tf.cast(tf.shape(x)[1], DTYPE)
    var = tf.exp(log_var)
    mahal = tf.reduce_sum((x - mean) ** 2 / var[tf.newaxis, :], axis=1)
    log_det = tf.reduce_sum(log_var)
    return -0.5 * (d * _LOG_2PI + log_det + mahal)


def log_gaussian_full(x, mean, cov_inv, log_det_cov):
    """Log N(x; mean, cov) given precision matrix and log-determinant.

    Parameters
    ----------
    x : tf.Tensor [N, d]
    mean : tf.Tensor [N, d] or [d]
    cov_inv : tf.Tensor [d, d]
    log_det_cov : tf.Tensor scalar

    Returns
    -------
    tf.Tensor [N]
    """
    d = tf.cast(tf.shape(x)[1], DTYPE)
    diff = x - mean
    mahal = tf.reduce_sum(diff * tf.linalg.matvec(cov_inv, diff), axis=1)
    return -0.5 * (d * _LOG_2PI + log_det_cov + mahal)


def dpf_elbo(particles_init, ys, ssm, proposal_fn, resampler, rng,
             stop_grad_resample=False):
    """Estimate the ELBO by running a DPF with a given proposal.

    Parameters
    ----------
    particles_init : tf.Tensor [N, d_x]
        Initial particles (sampled from prior).
    ys : tf.Tensor [T, d_y]
        Observations.
    ssm : object
        State-space model with attributes A, H, Q, R, Q_inv, R_inv,
        log_det_Q, log_det_R and methods f_batch, h_batch.
    proposal_fn : callable
        (particles, y_t) -> (x_new, m_prop, log_q)
        Returns sampled particles, proposal mean, and log-proposal density.
    resampler : ResamplerBase
        Resampling strategy.
    rng : tf.random.Generator
    stop_grad_resample : bool
        If True, detach gradients after resampling.

    Returns
    -------
    elbo : tf.Tensor scalar
        ELBO estimate: sum_t log(1/N sum_i w_t^i).
    mean_ess : tf.Tensor scalar
        Mean effective sample size across time steps.
    """
    T = ys.shape[0]
    N = tf.shape(particles_init)[0]
    N_f = tf.cast(N, DTYPE)

    particles = particles_init
    log_weights = tf.zeros(N, dtype=DTYPE)
    elbo = tf.constant(0.0, dtype=DTYPE)
    total_ess = tf.constant(0.0, dtype=DTYPE)

    for t in range(T):
        y_t = ys[t]

        # Proposal: sample x_new and compute log q(x_new | x_prev, y_t)
        x_new, log_q = proposal_fn(particles, y_t, rng)

        # Transition: log p(x_new | x_prev)
        Ax = ssm.f_batch(particles)
        log_trans = log_gaussian_full(x_new, Ax, ssm.Q_inv, ssm.log_det_Q)

        # Observation: log p(y_t | x_new)
        Hx = ssm.h_batch(x_new)
        log_obs = log_gaussian_full(
            tf.tile(y_t[tf.newaxis, :], [N, 1]), Hx, ssm.R_inv, ssm.log_det_R)

        # Importance weight increment
        log_w_inc = log_obs + log_trans - log_q

        # ELBO increment: log(1/N * sum w_t^i)
        log_w_total = log_weights + log_w_inc
        elbo = elbo + tf.reduce_logsumexp(log_w_total) - tf.math.log(N_f)

        # ESS
        log_weights = log_w_total - tf.reduce_max(log_w_total)
        w = tf.nn.softmax(log_weights)
        total_ess = total_ess + 1.0 / tf.reduce_sum(w ** 2)

        # Resampling
        if resampler is not None:
            p_new, lw_new = resampler.apply(log_weights, x_new, rng)
            if stop_grad_resample:
                particles = tf.stop_gradient(p_new)
                log_weights = tf.stop_gradient(lw_new)
            else:
                particles = p_new
                log_weights = lw_new
        else:
            particles = x_new

    return elbo, total_ess / tf.cast(T, DTYPE)
