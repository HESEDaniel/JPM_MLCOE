"""
Numerical utilities for particle flow algorithms.

Contains:
- Lambda schedule generators for flow integration
"""
import numpy as np


def exponential_lambda_schedule(n_steps: int = 29, ratio: float = 1.2) -> np.ndarray:
    """
    Generate exponentially spaced lambda positions for particle flow.

    As specified in Li et al. (2017):
    - N = 29 exponentially spaced step sizes
    - Constant ratio q = eps_j / eps_{j-1} = 1.2
    - Initial step size: eps_1 = (1-q) / (1-q^N)
    - This ensures sum of step sizes = 1

    Parameters
    ----------
    n_steps : int
        Number of flow steps (default: 29)
    ratio : float
        Ratio between consecutive step sizes (default: 1.2)

    Returns
    -------
    lambda_positions : ndarray [n_steps + 1]
        Lambda positions from 0 to 1 with exponential spacing
        lambda_positions[0] = 0, lambda_positions[-1] = 1
    """
    q = ratio
    N = n_steps

    # Initial step size: eps_1 = (1-q) / (1-q^N)
    eps_1 = (1 - q) / (1 - q**N)

    # Step sizes: eps_j = eps_1 * q^{j-1} for j = 1, ..., N
    step_sizes = eps_1 * (q ** np.arange(N))

    # Lambda positions are cumulative sums starting from 0
    lambda_positions = np.zeros(N + 1)
    lambda_positions[1:] = np.cumsum(step_sizes)

    # Ensure last position is exactly 1 (numerical precision)
    lambda_positions[-1] = 1.0

    return lambda_positions
