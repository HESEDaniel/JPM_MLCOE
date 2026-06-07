"""Alternative latent x_t generators for the form-misspecification experiment.

The filter assumes a scalar AR(1)-Gaussian latent. These produce a true x_t that is
NOT AR(1), to measure how the AR(1) bootstrap PF degrades under form-level mismatch.
Each process is used at its natural scale (no post-hoc rescaling): the filter fits its
own best AR(1) parameters to whatever series is generated, exactly as a deployment would.

  ar1     : AR(1) baseline (self-check; matches the oracle case).
  ar2     : cyclical AR(2) with complex roots (a pseudo-period AR(1) cannot represent).
  regime  : 2-state Markov-switching mean (discrete jumps the Gaussian AR(1) smooths over).

The two autoregressive processes start from zero and discard a burn-in so the kept
samples are stationary. The regime process starts from its stationary 50/50 state and
needs no burn-in.
"""
from __future__ import annotations

import numpy as np


def ar1_process(T, rng, mu=0.0, phi=0.9, sigma=0.5, burn=100):
    """Generate a scalar AR(1)-Gaussian latent series.

    Starts from zero and discards a burn-in so the kept samples are stationary.
    This is the baseline self-check process that matches the oracle AR(1) case.

    Parameters
    ----------
    T : int
        Number of stationary samples to return.
    rng : numpy.random.Generator
        Random generator for the Gaussian innovations.
    mu : float
        Unconditional mean offset of the AR(1) recursion.
    phi : float
        Autoregressive coefficient.
    sigma : float
        Standard deviation of the Gaussian innovations.
    burn : int
        Number of leading burn-in samples to discard.

    Returns
    -------
    numpy.ndarray
        A (T,) float32 array of the stationary latent series.
    """
    n = T + burn
    x = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        x[t] = mu + phi * x[t - 1] + rng.normal(0.0, sigma)
    return x[burn:].astype(np.float32)


def ar2_process(T, rng, phi1=1.4, phi2=-0.8, sigma=0.5, burn=100):
    """Generate a cyclical AR(2)-Gaussian latent series.

    Roots are complex when phi1^2 + 4*phi2 < 0 (here 1.96 - 3.2 < 0), giving a
    damped pseudo-period that a pseudo-period AR(1) cannot represent. The process
    is stationary since phi1+phi2<1, phi2-phi1<1, and |phi2|<1. Starts from zero
    and discards a burn-in so the kept samples are stationary.

    Parameters
    ----------
    T : int
        Number of stationary samples to return.
    rng : numpy.random.Generator
        Random generator for the Gaussian innovations.
    phi1 : float
        First-lag autoregressive coefficient.
    phi2 : float
        Second-lag autoregressive coefficient.
    sigma : float
        Standard deviation of the Gaussian innovations.
    burn : int
        Number of leading burn-in samples to discard.

    Returns
    -------
    numpy.ndarray
        A (T,) float32 array of the stationary latent series.
    """
    n = T + burn
    x = np.zeros(n, dtype=np.float64)
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + rng.normal(0.0, sigma)
    return x[burn:].astype(np.float32)


def regime_switching_process(T, rng, mus=(-1.0, 1.0), p_stay=0.95, sigma=0.3):
    """Generate a 2-state Markov-switching-mean latent series.

    The initial regime is drawn from the stationary 50/50 distribution, so no
    burn-in is needed. Persistent regimes (~1/(1-p_stay) steps) with small
    within-regime Gaussian noise produce discrete level jumps that the Gaussian
    AR(1) filter smooths over.

    Parameters
    ----------
    T : int
        Number of samples to return.
    rng : numpy.random.Generator
        Random generator for the regime transitions and Gaussian noise.
    mus : tuple of float
        Mean level of each of the two regimes.
    p_stay : float
        Probability of staying in the current regime at each step.
    sigma : float
        Standard deviation of the within-regime Gaussian noise.

    Returns
    -------
    numpy.ndarray
        A (T,) float32 array of the latent series.
    """
    x = np.empty(T, dtype=np.float64)
    s = int(rng.random() < 0.5)
    for t in range(T):
        if rng.random() > p_stay:
            s = 1 - s
        x[t] = mus[s] + rng.normal(0.0, sigma)
    return x.astype(np.float32)


PROCESSES = {
    "ar1": ar1_process,
    "ar2": ar2_process,
    "regime": regime_switching_process,
}


def generate(process: str, T: int, rng) -> np.ndarray:
    """Dispatch to a named latent process and return its sampled series.

    Parameters
    ----------
    process : str
        Name of the process to generate; one of the keys in ``PROCESSES``.
    T : int
        Number of samples to return.
    rng : numpy.random.Generator
        Random generator passed to the selected process.

    Returns
    -------
    numpy.ndarray
        A (T,) float32 array of the true latent series at its natural scale.
    """
    if process not in PROCESSES:
        raise ValueError(f"unknown x_t process {process!r}; choose from {list(PROCESSES)}")
    return PROCESSES[process](T, rng)


def fit_best_ar1(x: np.ndarray):
    """Fit the best AR(1) parameters by OLS of x_t on x_{t-1}.

    The slope is clipped to keep |phi|<1 (a stationarity guard), and a tiny
    floor is added to the residual standard deviation to keep it strictly
    positive.

    Parameters
    ----------
    x : numpy.ndarray
        The latent series to fit.

    Returns
    -------
    tuple of float
        The fitted ``(mu, phi, sigma)`` AR(1) parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    x0, x1 = x[:-1], x[1:]
    var0 = np.var(x0)
    phi = float(np.cov(x0, x1, bias=True)[0, 1] / var0) if var0 > 1e-8 else 0.0
    phi = float(np.clip(phi, -0.99, 0.99))   # |phi|<1 guard
    mu = float(x1.mean() - phi * x0.mean())
    resid = x1 - (mu + phi * x0)
    sigma = float(resid.std() + 1e-6)
    return mu, phi, sigma
