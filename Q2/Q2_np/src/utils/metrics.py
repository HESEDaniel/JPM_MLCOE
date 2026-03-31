"""
Metrics for evaluating filter performance.
"""
import numpy as np


def compute_mse(estimated, true):
    """
    Compute Mean Squared Error.

    Parameters
    ----------
    estimated : ndarray
        Estimated values
    true : ndarray
        True values

    Returns
    -------
    float
        Mean squared error
    """
    return np.mean((estimated - true)**2)


def compute_rmse(estimated, true):
    """
    Compute Root Mean Squared Error.

    Parameters
    ----------
    estimated : ndarray
        Estimated values
    true : ndarray
        True values

    Returns
    -------
    float
        Root mean squared error
    """
    return np.sqrt(compute_mse(estimated, true))


def compute_nees(m_filt, P_filt, xs, regularize=1e-8):
    """
    Compute Normalized Estimation Error Squared (NEES).

    NEES = (x - m)' * P^{-1} * (x - m)

    For a consistent filter, NEES should follow chi-squared(n_x) distribution.

    Parameters
    ----------
    m_filt : ndarray [T, n_x]
        Filtered means
    P_filt : ndarray [T, n_x, n_x]
        Filtered covariances
    xs : ndarray [T, n_x]
        True states
    regularize : float
        Small value added to diagonal for numerical stability

    Returns
    -------
    ndarray [T]
        NEES values at each time step
    """
    T = m_filt.shape[0]
    n_x = m_filt.shape[1]
    nees = np.zeros(T)

    for t in range(T):
        error = xs[t] - m_filt[t]
        P_reg = P_filt[t] + regularize * np.eye(n_x)
        try:
            nees[t] = error @ np.linalg.solve(P_reg, error)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for singular matrices
            nees[t] = error @ np.linalg.lstsq(P_reg, error, rcond=None)[0]

    return nees


def stability_summary(cond_nums, mse=None):
    """
    Generate summary statistics for numerical stability metrics.

    Parameters
    ----------
    cond_nums : ndarray
        Condition numbers
    mse : float, optional
        Mean squared error

    Returns
    -------
    dict
        Summary statistics
    """
    summary = {
        'mean_cond': np.mean(cond_nums),
        'max_cond': np.max(cond_nums),
    }
    if mse is not None:
        summary['mse'] = mse
    return summary


def compute_symmetry_error(P_filt):
    """
    Compute symmetry error ||P - P'||_F / ||P||_F over all time steps.

    Parameters
    ----------
    P_filt : ndarray [T, n_x, n_x]
        Covariance matrices

    Returns
    -------
    ndarray [T]
        Relative symmetry error at each time step
    """
    T = P_filt.shape[0]
    sym_err = np.zeros(T)
    for t in range(T):
        P = P_filt[t]
        norm_P = np.linalg.norm(P, 'fro')
        if norm_P > 0:
            sym_err[t] = np.linalg.norm(P - P.T, 'fro') / norm_P
        else:
            sym_err[t] = 0.0
    return sym_err


def compute_min_eigenvalues(P_filt):
    """
    Compute minimum eigenvalue of P at each time step.

    Negative values indicate loss of positive semi-definiteness.

    Parameters
    ----------
    P_filt : ndarray [T, n_x, n_x]
        Covariance matrices

    Returns
    -------
    ndarray [T]
        Minimum eigenvalue at each time step
    """
    T = P_filt.shape[0]
    min_eig = np.zeros(T)
    for t in range(T):
        min_eig[t] = np.linalg.eigvalsh(P_filt[t]).min()
    return min_eig


def compute_nis(innovations, S_innov):
    """
    Compute Normalized Innovation Squared (NIS).

    NIS = (y - Hx)' S^{-1} (y - Hx)

    For a consistent filter, NIS should follow chi-squared(n_y) distribution.

    Parameters
    ----------
    innovations : ndarray [T, n_y]
        Innovation vectors (y - Hx)
    S_innov : ndarray [T, n_y, n_y]
        Innovation covariances

    Returns
    -------
    ndarray [T]
        NIS values at each time step
    """
    T = innovations.shape[0]
    nis = np.zeros(T)
    for t in range(T):
        v = innovations[t]
        S = S_innov[t]
        nis[t] = v @ np.linalg.solve(S, v)
    return nis

