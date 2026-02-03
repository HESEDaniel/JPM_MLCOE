"""
Visualization utilities for particle filter results.

Functions for plotting:
- OMAT error comparisons
- Effective Sample Size (ESS)
- Performance boxplots
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# Default color schemes for common algorithms
DEFAULT_COLORS = {
    'EKF': '#1f77b4', 'UKF': '#ff7f0e',
    'BPF': '#2ca02c', 'BPF(100K)': '#1f77b4', 'BPF(1M)': '#ff7f0e',
    'EDH': '#d62728', 'LEDH': '#9467bd',
    'PF-PF(EDH)': '#8c564b', 'PF-PF(LEDH)': '#e377c2'
}

DEFAULT_LINESTYLES = {
    'EKF': '-', 'UKF': '-',
    'BPF': ':', 'BPF(100K)': ':', 'BPF(1M)': ':',
    'EDH': '--', 'LEDH': '--',
    'PF-PF(EDH)': '-', 'PF-PF(LEDH)': '-'
}


def plot_omat_comparison(
    omat_results: Dict[str, np.ndarray],
    save_path: str,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    title: str = 'Average OMAT Error Over Time',
    xlabel: str = 'Time Step',
    ylabel: str = 'OMAT Error (m)',
    figsize: tuple = (10, 5)
) -> None:
    """
    Plot OMAT error time-series comparison across algorithms.

    Parameters
    ----------
    omat_results : dict
        Dictionary mapping algorithm names to OMAT arrays [T]
    save_path : str
        Path to save the figure
    colors : dict, optional
        Color mapping for algorithms. Uses defaults if None.
    linestyles : dict, optional
        Linestyle mapping for algorithms. Uses defaults if None.
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    """
    if colors is None:
        colors = DEFAULT_COLORS
    if linestyles is None:
        linestyles = DEFAULT_LINESTYLES

    fig, ax = plt.subplots(figsize=figsize)

    for name, omat in omat_results.items():
        mean_omat = np.mean(omat)
        ax.plot(omat, label=f'{name} (mean={mean_omat:.2f})',
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ess_comparison(
    ess_results: Dict[str, np.ndarray],
    N_particles: int,
    save_path: str,
    resample_threshold: float = 0.5,
    colors: Optional[Dict[str, str]] = None,
    title: str = 'Effective Sample Size Over Time',
    xlabel: str = 'Time Step',
    ylabel: str = 'ESS',
    figsize: tuple = (10, 4)
) -> None:
    """
    Plot ESS time-series comparison across particle filter algorithms.

    Parameters
    ----------
    ess_results : dict
        Dictionary mapping algorithm names to ESS arrays [T]
    N_particles : int
        Number of particles (for threshold line)
    save_path : str
        Path to save the figure
    resample_threshold : float
        Resampling threshold as fraction of N_particles
    colors : dict, optional
        Color mapping for algorithms. Uses defaults if None.
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    """
    if colors is None:
        colors = DEFAULT_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    max_ess = max(np.max(ess) for ess in ess_results.values()) if ess_results else N_particles

    for name, ess in ess_results.items():
        ax.plot(ess, label=f'{name} (mean={np.mean(ess):.1f})',
                color=colors.get(name, 'gray'), linewidth=1.5)

    # Resample threshold line
    threshold = N_particles * resample_threshold
    if threshold <= max_ess * 1.1:
        ax.axhline(threshold, color='gray', linestyle='--', alpha=0.5,
                   label=f'Resample threshold ({threshold:.0f})')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, max_ess * 1.1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_omat_boxplot(
    all_omat: Dict[str, List[np.ndarray]],
    save_path: str,
    algorithm_order: Optional[List[str]] = None,
    title: str = 'Boxplots of Average OMAT Errors',
    xlabel: str = 'Algorithm',
    ylabel: str = 'Average OMAT Error (m)',
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot boxplots of per-trial average OMAT errors.

    Parameters
    ----------
    all_omat : dict
        Dictionary mapping algorithm names to lists of OMAT arrays
        Each OMAT array has shape [T] with OMAT at each time step
    save_path : str
        Path to save the figure
    algorithm_order : list, optional
        Order of algorithms in plot. If None, uses default order.
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    """
    # Compute mean OMAT per trial for each algorithm
    data_to_plot = {}
    for algo, omat_list in all_omat.items():
        mean_omat_per_trial = [np.mean(omat) for omat in omat_list]
        data_to_plot[algo] = mean_omat_per_trial

    # Determine plot order
    if algorithm_order is None:
        # Default order matching Li17 paper style
        algorithm_order = ['PF-PF(LEDH)', 'PF-PF(EDH)', 'LEDH', 'EDH',
                           'EKF', 'UKF', 'BPF', 'BPF(100K)', 'BPF(1M)']

    plot_order = [algo for algo in algorithm_order if algo in data_to_plot]
    # Add any algorithms not in default order
    for algo in data_to_plot:
        if algo not in plot_order:
            plot_order.append(algo)

    plot_data = [data_to_plot[algo] for algo in plot_order]
    labels = plot_order

    fig, ax = plt.subplots(figsize=figsize)

    ax.boxplot(plot_data, tick_labels=labels, patch_artist=True,
               showmeans=False, showfliers=True,
               medianprops=dict(color='red', linewidth=1.5),
               boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=1),
               whiskerprops=dict(color='black', linestyle='--', linewidth=1),
               capprops=dict(color='black', linewidth=1),
               flierprops=dict(marker='o', markerfacecolor='red',
                               markeredgecolor='red', markersize=4, alpha=0.6))

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_weight_distributions(
    weights_history: np.ndarray,
    time_indices: List[int],
    N_particles: int,
    filter_name: str = 'SIS',
    save_path: Optional[str] = None,
    n_bins: int = 50,
    figsize_per_plot: tuple = (5, 4)
) -> None:
    """
    Plot weight distribution histograms at specified time steps.

    Parameters
    ----------
    weights_history : ndarray [T, N]
        Weights at each time step
    time_indices : list of int
        Time indices to plot
    N_particles : int
        Number of particles
    filter_name : str
        Name for title
    save_path : str, optional
        Path to save figure
    n_bins : int
        Number of histogram bins
    figsize_per_plot : tuple
        Figure size per subplot
    """
    import os

    n_plots = len(time_indices)
    fig, axes = plt.subplots(1, n_plots,
                              figsize=(figsize_per_plot[0] * n_plots, figsize_per_plot[1]))
    if n_plots == 1:
        axes = [axes]

    for idx, t in enumerate(time_indices):
        ax = axes[idx]
        weights = weights_history[t]

        ax.hist(weights, bins=n_bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')
        ax.set_title(f'n={t}')

        ess = 1.0 / np.sum(weights**2)
        ax.text(0.95, 0.95, f'ESS: {ess:.1f}', transform=ax.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{filter_name}: Weight Distributions (N={N_particles})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(save_path)}")
    plt.close()
