"""
Visualization functions for Kalman filter family results.
"""
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_kalman_filter(T, xs, ys, m_filt, P_filt, save_path=None, title="Kalman Filter"):
    """
    Plot Kalman filter results.

    Parameters
    ----------
    T : int
        Number of time steps
    xs : ndarray [T, n_x]
        True states
    ys : ndarray [T, n_y]
        Observations
    m_filt : ndarray [T, n_x]
        Filtered means
    P_filt : ndarray [T, n_x, n_x]
        Filtered covariances
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    t = np.arange(T)
    n_x = xs.shape[1] if xs.ndim > 1 else 1

    if n_x == 1:
        xs = xs.reshape(-1, 1)
        m_filt = m_filt.reshape(-1, 1)

    fig, axes = plt.subplots(n_x, 1, figsize=(12, 4*n_x))
    if n_x == 1:
        axes = [axes]

    for i in range(n_x):
        ax = axes[i]
        std_filt = np.sqrt(P_filt[:, i, i])

        ax.plot(t, xs[:, i], 'k-', linewidth=2, label='True State', alpha=0.8)
        ax.plot(t, m_filt[:, i], 'b--', linewidth=1.5, label='Filter Mean', alpha=0.8)
        ax.fill_between(t, m_filt[:, i] - 2*std_filt, m_filt[:, i] + 2*std_filt,
                        alpha=0.2, color='blue', label='+/-2sigma')

        ax.set_xlabel('Time')
        ax.set_ylabel(f'State {i+1}')
        ax.set_title(f'{title} - State {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_filter_estimate_with_bands(t, xs_true, m_filt, P_filt, filter_name,
                                     state_idx=0, n_sigma=1.0,
                                     diagnostic_data=None,
                                     diagnostic_label='Absolute Error',
                                     diagnostic_log_scale=True,
                                     save_path=None, figsize=(12, 8)):
    """
    Plot filter estimates with uncertainty bands and optional diagnostic subplot.

    Parameters
    ----------
    t : ndarray [T]
        Time array
    xs_true : ndarray [T] or [T, n_x]
        True state(s)
    m_filt : ndarray [T, n_x]
        Filtered means
    P_filt : ndarray [T, n_x, n_x]
        Filtered covariances
    filter_name : str
        Name for title/legend
    state_idx : int
        Which state dimension to plot (default: 0)
    n_sigma : float
        Number of standard deviations for bands (default: 1.0)
    diagnostic_data : ndarray [T], optional
        Data for diagnostic subplot (e.g., ESS, condition number)
    diagnostic_label : str
        Y-axis label for diagnostic subplot
    diagnostic_log_scale : bool
        Use log scale for diagnostic subplot
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Handle 1D true states
    if xs_true.ndim == 1:
        xs = xs_true
    else:
        xs = xs_true[:, state_idx]

    x_filt = m_filt[:, state_idx]
    std_filt = np.sqrt(P_filt[:, state_idx, state_idx])

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # State estimation plot
    ax1 = axes[0]
    ax1.plot(t, xs, 'b+', markersize=6, label='True State', alpha=0.8)
    ax1.plot(t, x_filt, 'r-', linewidth=2, label='Filter Mean')
    ax1.fill_between(t, x_filt - n_sigma * std_filt, x_filt + n_sigma * std_filt,
                     alpha=0.3, color='red', label=f'+/- {n_sigma} S.D.')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('State')
    ax1.set_title(f'{filter_name} Estimate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Diagnostic subplot
    ax2 = axes[1]
    if diagnostic_data is not None:
        ax2.plot(t, diagnostic_data, 'b-', linewidth=2)
    else:
        ax2.plot(t, np.abs(x_filt - xs), 'b-', linewidth=2)
        diagnostic_label = 'Absolute Error'

    if diagnostic_log_scale:
        ax2.set_yscale('log')
    ax2.set_xlabel('Time')
    ax2.set_ylabel(diagnostic_label)
    ax2.set_title(f'{filter_name} Diagnostics')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(save_path)}")
    plt.close()


def plot_multi_filter_comparison(t, xs_true, filter_results, state_idx=0,
                                  n_sigma=1.0, suptitle=None, save_path=None,
                                  figsize=(14, 10), ncols=2):
    """
    Plot multi-panel comparison of multiple filters with uncertainty bands.

    Parameters
    ----------
    t : ndarray [T]
        Time array
    xs_true : ndarray [T] or [T, n_x]
        True state(s)
    filter_results : dict
        Dictionary mapping filter names to (m_filt, P_filt, metric) tuples
        where metric can be condition number, ESS, etc.
    state_idx : int
        Which state dimension to plot (for multi-dimensional states)
    n_sigma : float
        Number of standard deviations for bands
    suptitle : str, optional
        Super title for figure
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    ncols : int
        Number of columns in subplot grid
    """
    # Handle 1D true states
    if xs_true.ndim == 1:
        xs = xs_true
    else:
        xs = xs_true[:, state_idx]

    n_filters = len(filter_results)
    nrows = (n_filters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_filters == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, data) in enumerate(filter_results.items()):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        m_filt, P_filt, metric = data
        x_filt = m_filt[:, state_idx]
        std_filt = np.sqrt(P_filt[:, state_idx, state_idx])

        ax.plot(t, xs, 'b+', markersize=4, label='True', alpha=0.7)
        ax.plot(t, x_filt, 'r-', linewidth=2, label='Estimate')
        ax.fill_between(t, x_filt - n_sigma * std_filt, x_filt + n_sigma * std_filt,
                        alpha=0.3, color='red', label=f'+/-{n_sigma}sigma')

        mse = np.mean((x_filt - xs)**2)
        ax.set_title(f'{name}\nMSE={mse:.4f}, Mean Cond#={np.mean(metric):.1f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('State')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_filters, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.basename(save_path)}")
    plt.close()


def plot_stability_analysis(T, xs, m_joseph, m_std, cond_joseph, cond_std,
                            save_path=None):
    """
    Plot stability analysis comparing Joseph vs standard covariance update.

    Parameters
    ----------
    T : int
        Number of time steps
    xs : ndarray [T, n_x]
        True states
    m_joseph, m_std : ndarray [T, n_x]
        Filtered means (Joseph and standard)
    cond_joseph, cond_std : ndarray [T]
        Condition numbers
    save_path : str, optional
        Path to save figure
    """
    t = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Condition numbers
    ax1 = axes[0]
    ax1.semilogy(t, cond_joseph, 'b-', linewidth=1.5, label='Joseph', alpha=0.8)
    ax1.semilogy(t, cond_std, 'r--', linewidth=1.5, label='Standard', alpha=0.8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Condition Number')
    ax1.set_title('Condition Number Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Estimation error
    ax2 = axes[1]
    err_joseph = np.abs(m_joseph[:, 0] - xs[:, 0])
    err_std = np.abs(m_std[:, 0] - xs[:, 0])
    ax2.plot(t, err_joseph, 'b-', linewidth=1.5, label='Joseph', alpha=0.8)
    ax2.plot(t, err_std, 'r--', linewidth=1.5, label='Standard', alpha=0.8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Estimation Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Stability Analysis: Joseph vs Standard Covariance Update',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_filter_comparison(t, xs, results, title, save_path=None):
    """Plot 2x2 grid comparing multiple filters."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (name, (m, P, _)) in zip(axes.flat, results.items()):
        std = np.sqrt(P[:, 0, 0])
        ax.plot(t, xs, 'k-', lw=1.5, alpha=0.7, label='True')
        ax.plot(t, m[:, 0], 'r-', lw=1.5, label='Estimate')
        ax.fill_between(t, m[:, 0] - 2*std, m[:, 0] + 2*std, alpha=0.2, color='red')
        mse = np.mean((m[:, 0] - xs)**2)
        ax.set_title(f'{name} (MSE={mse:.4f})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_linearization_analysis(t, xs, m_ekf, lin_error, jacobian, hessian, save_path=None):
    """Plot EKF linearization error analysis (4 subplots)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(t, np.abs(xs - m_ekf[:, 0]), 'b-', lw=1.5)
    axes[0, 0].set_ylabel('|x_true - x_estimate|')
    axes[0, 0].set_title('State Estimation Error')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(t, lin_error + 1e-10, 'r-', lw=1.5)
    axes[0, 1].set_ylabel('Linearization Error')
    axes[0, 1].set_title('Taylor Approximation Error')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, jacobian, 'g-', lw=1.5)
    axes[1, 0].set_ylabel('H(x) = dh/dx')
    axes[1, 0].set_title('Jacobian Variation')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t, hessian, 'm-', lw=1.5)
    axes[1, 1].set_ylabel("d^2h/dx^2")
    axes[1, 1].set_title('Curvature (EKF truncation source)')
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Time')

    plt.suptitle('EKF Linearization Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sigma_point_analysis(sigma_analysis_list, P_values, save_path=None):
    """Plot UKF sigma point transformation analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, (sa, P) in enumerate(zip(sigma_analysis_list, P_values)):
        axes[0, 0].scatter(sa['sigma_x'], [i]*len(sa['sigma_x']), s=50, label=f'P={P:.1f}')
        axes[0, 1].scatter(sa['sigma_h'], [i]*len(sa['sigma_h']), s=50, label=f'P={P:.1f}')

    axes[0, 0].set_xlabel('State x')
    axes[0, 0].set_title('Sigma Points (Gaussian)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('h(x)')
    axes[0, 1].set_title('Transformed Points (asymmetric)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    biases = [sa['mean_bias'] for sa in sigma_analysis_list]
    axes[1, 0].bar(range(len(P_values)), biases, tick_label=[f'P={p:.1f}' for p in P_values])
    axes[1, 0].axhline(0, color='k', ls='--', lw=1)
    axes[1, 0].set_ylabel('UKF Mean - True Mean')
    axes[1, 0].set_title("Mean Bias (Jensen's Inequality)")
    axes[1, 0].grid(True, alpha=0.3)

    ratios = [sa['var_ratio'] for sa in sigma_analysis_list]
    axes[1, 1].bar(range(len(P_values)), ratios, tick_label=[f'P={p:.1f}' for p in P_values])
    axes[1, 1].axhline(1, color='k', ls='--', lw=1)
    axes[1, 1].set_ylabel('UKF Var / True Var')
    axes[1, 1].set_title('Variance Ratio (<1 = underestimate)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('UKF Sigma Point Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_nees_comparison(t, nees_dict, save_path=None):
    """Plot NEES comparison for filter consistency."""
    n = len(nees_dict)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (name, nees) in zip(axes.flat[:n], nees_dict.items()):
        ax.plot(t, nees, 'b-', lw=1, alpha=0.7)
        ax.axhline(1, color='r', ls='--', lw=2, label='Consistent (NEES=1)')
        ax.axhline(np.mean(nees), color='g', ls='-', lw=2, label=f'Mean={np.mean(nees):.2f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('NEES')
        ax.set_title(name)
        ax.set_ylim([0, min(20, np.percentile(nees, 99))])
        ax.legend()
        ax.grid(True, alpha=0.3)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    plt.suptitle('NEES (should be ~1 if consistent)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison_side_by_side(t, xs, results_top, results_bottom,
                                        top_label, bottom_label, save_path=None):
    """Side-by-side comparison of two observation models."""
    n_cols = len(results_top)
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8))

    for i, (name, (m, P, _)) in enumerate(results_top.items()):
        ax = axes[0, i]
        std = np.sqrt(P[:, 0, 0])
        ax.plot(t, xs, 'k-', lw=1, alpha=0.7)
        ax.plot(t, m[:, 0], 'b-', lw=1.5)
        ax.fill_between(t, m[:, 0] - 2*std, m[:, 0] + 2*std, alpha=0.2, color='blue')
        mse = np.mean((m[:, 0] - xs)**2)
        ax.set_title(f'{name}\nMSE={mse:.4f}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel(top_label)

    for i, (name, (m, P, _)) in enumerate(results_bottom.items()):
        ax = axes[1, i]
        std = np.sqrt(P[:, 0, 0])
        ax.plot(t, xs, 'k-', lw=1, alpha=0.7)
        ax.plot(t, m[:, 0], 'r-', lw=1.5)
        ax.fill_between(t, m[:, 0] - 2*std, m[:, 0] + 2*std, alpha=0.2, color='red')
        mse = np.mean((m[:, 0] - xs)**2)
        ax.set_title(f'MSE={mse:.4f}')
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel(bottom_label)

    plt.suptitle(f'{top_label} vs {bottom_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
