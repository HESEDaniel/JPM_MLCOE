"""Bootstrap Particle Filter implementation."""
import numpy as np


def particle_filter(f, h, Q_sampler, log_likelihood, m0, P0, ys,
                    N_particles=1000, resample_threshold=0.5, rng=None):
    """
    Bootstrap Particle Filter (BPF).
    Run Sequential Importance Sampling (SIS) with resample_threshold=0.0.

    Parameters
    ----------
    f : callable
        State transition: f(x) -> x_next (deterministic part)
    h : callable
        Observation: h(x) -> y
    Q_sampler : callable
        Process noise sampler: Q_sampler(rng, N) -> [N, n_x]
    log_likelihood : callable
        Log-likelihood: log_likelihood(y, particles) -> [N]
    m0 : ndarray [n_x]
        Initial mean
    P0 : ndarray [n_x, n_x]
        Initial covariance
    ys : ndarray [T, n_y]
        Observations
    N_particles : int
        Number of particles
    resample_threshold : float
        Resample when ESS < threshold * N_particles
    rng : np.random.Generator

    Returns
    -------
    m_filt : ndarray [T, n_x]
    P_filt : ndarray [T, n_x, n_x]
    ess : ndarray [T]
        Effective sample size
    resample_count : int
    """
    if rng is None:
        rng = np.random.default_rng()

    T, n_x = ys.shape[0], len(m0)
    N = N_particles
    particles = rng.multivariate_normal(m0, P0, size=N)
    log_w = np.zeros(N)

    m_filt = np.zeros((T, n_x))
    P_filt = np.zeros((T, n_x, n_x))
    ess = np.zeros(T)
    resample_count = 0

    for t in range(T):
        # Predict
        noise = Q_sampler(rng, N)
        try:
            particles = f(particles) + noise
        except (ValueError, IndexError):
            particles = np.array([f(p) for p in particles]) + noise

        # Update weights
        log_w = log_likelihood(ys[t], particles)
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w /= w.sum()

        # ESS and resample
        ess[t] = 1.0 / (w ** 2).sum()
        if ess[t] < resample_threshold * N:
            idx = systematic_resample(w, rng)
            particles = particles[idx]
            w = np.ones(N) / N
            resample_count += 1

        # Estimate
        m_filt[t] = w @ particles
        diff = particles - m_filt[t]
        P_filt[t] = np.einsum('i,ij,ik->jk', w, diff, diff)

    return m_filt, P_filt, ess, resample_count


def systematic_resample(w, rng):
    """Systematic resampling (low variance)."""
    N = len(w)
    cumsum = np.cumsum(w)
    u = rng.uniform(0, 1.0 / N) + np.arange(N) / N
    return np.clip(np.searchsorted(cumsum, u), 0, N - 1)
