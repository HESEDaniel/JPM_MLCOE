"""
Visualization utilities for tracking and trajectory plots.

Functions for plotting:
- Covariance ellipses
- 2D trajectories
- Contour plots
"""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_covariance_ellipse(
    ax: plt.Axes,
    mean: np.ndarray,
    cov: np.ndarray,
    n_std: float = 2.0,
    **kwargs
) -> Ellipse:
    """
    Plot covariance ellipse on given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    mean : ndarray [2]
        Center of ellipse (x, y)
    cov : ndarray [2, 2]
        2x2 covariance matrix
    n_std : float
        Number of standard deviations for ellipse size
    **kwargs
        Additional arguments passed to matplotlib.patches.Ellipse
        (e.g., fill, color, alpha, linewidth, linestyle)

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        The created ellipse patch
    """
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Compute angle (rotation from x-axis)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Compute width and height (2 * n_std * sqrt(eigenvalue))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(mean, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def plot_contours(
    ax: plt.Axes,
    mean: np.ndarray,
    cov: np.ndarray,
    levels: List[float] = [1, 2, 3],
    **kwargs
) -> List[Ellipse]:
    """
    Plot multiple confidence ellipses as contours.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    mean : ndarray [2]
        Center of ellipses
    cov : ndarray [2, 2]
        2x2 covariance matrix
    levels : list of float
        Standard deviation levels for contours (e.g., [1, 2, 3] for 1sigma, 2sigma, 3sigma)
    **kwargs
        Additional arguments passed to plot_covariance_ellipse

    Returns
    -------
    ellipses : list of Ellipse
        Created ellipse patches
    """
    ellipses = []
    for level in levels:
        ellipse = plot_covariance_ellipse(
            ax, mean, cov, n_std=level,
            fill=False, **kwargs
        )
        ellipses.append(ellipse)
    return ellipses


def plot_trajectory_2d(
    ax: plt.Axes,
    trajectory: np.ndarray,
    label: Optional[str] = None,
    color: str = 'blue',
    linestyle: str = '-',
    linewidth: float = 1.5,
    marker_start: str = 'o',
    marker_end: str = 'x',
    marker_size: float = 80,
    **kwargs
) -> None:
    """
    Plot a 2D trajectory with start/end markers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    trajectory : ndarray [T, 2]
        Trajectory points (x, y)
    label : str, optional
        Label for legend
    color : str
        Line color
    linestyle : str
        Line style
    linewidth : float
        Line width
    marker_start : str
        Marker for start point
    marker_end : str
        Marker for end point
    marker_size : float
        Size of markers
    **kwargs
        Additional arguments passed to ax.plot
    """
    ax.plot(trajectory[:, 0], trajectory[:, 1],
            color=color, linestyle=linestyle, linewidth=linewidth,
            label=label, **kwargs)

    # Start marker
    if marker_start:
        ax.scatter(trajectory[0, 0], trajectory[0, 1],
                   color=color, marker=marker_start, s=marker_size, zorder=10)

    # End marker
    if marker_end:
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                   color=color, marker=marker_end, s=marker_size, zorder=10)


def plot_particle_cloud(
    ax: plt.Axes,
    particles: np.ndarray,
    color: str = 'blue',
    alpha: float = 0.5,
    size: float = 20,
    label: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot a cloud of particles.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    particles : ndarray [N, 2]
        Particle positions (x, y)
    color : str
        Particle color
    alpha : float
        Transparency
    size : float
        Marker size
    label : str, optional
        Label for legend
    **kwargs
        Additional arguments passed to ax.scatter
    """
    ax.scatter(particles[:, 0], particles[:, 1],
               c=color, s=size, alpha=alpha, label=label, **kwargs)


def plot_trajectory_comparison(
    ax: plt.Axes,
    xs_true: np.ndarray,
    estimates: dict,
    pos_indices: tuple = (0, 2),
    sensor_pos: Optional[np.ndarray] = None,
    colors: Optional[dict] = None,
    linestyles: Optional[dict] = None,
    show_mse: bool = True,
    show_start_end: bool = True,
    true_label: str = 'True',
    true_color: str = 'black',
    true_linestyle: str = '-',
) -> None:
    """
    Plot 2D trajectory comparison between true and estimated paths.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    xs_true : ndarray [T, n_x]
        True states
    estimates : dict
        Dictionary mapping algorithm names to estimated states [T, n_x]
    pos_indices : tuple of int
        Indices for x and y position in state vector (default: (0, 2) for [x, vx, y, vy])
    sensor_pos : ndarray [n_sensors, 2], optional
        Sensor positions to mark
    colors : dict, optional
        Color mapping for algorithms
    linestyles : dict, optional
        Linestyle mapping for algorithms
    show_mse : bool
        Show MSE in legend labels
    show_start_end : bool
        Show start (o) and end (x) markers
    true_label : str
        Label for true trajectory
    true_color : str
        Color for true trajectory
    true_linestyle : str
        Linestyle for true trajectory
    """
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if colors is None:
        colors = {}
    if linestyles is None:
        linestyles = {}

    ix, iy = pos_indices
    true_pos = np.stack([xs_true[:, ix], xs_true[:, iy]], axis=1)

    # Plot true trajectory
    ax.plot(true_pos[:, 0], true_pos[:, 1], color=true_color, linestyle=true_linestyle,
            linewidth=2, label=true_label, alpha=0.8)
    if show_start_end:
        ax.scatter(true_pos[0, 0], true_pos[0, 1], color=true_color, marker='o', s=80, zorder=10)
        ax.scatter(true_pos[-1, 0], true_pos[-1, 1], color=true_color, marker='x', s=80, zorder=10)

    # Plot estimates
    for idx, (name, est) in enumerate(estimates.items()):
        est_pos = np.stack([est[:, ix], est[:, iy]], axis=1)
        color = colors.get(name, default_colors[idx % len(default_colors)])
        ls = linestyles.get(name, '--')

        if show_mse:
            mse = np.mean((est_pos - true_pos)**2)
            label = f'{name} (MSE={mse:.2f})'
        else:
            label = name

        ax.plot(est_pos[:, 0], est_pos[:, 1], color=color, linestyle=ls,
                linewidth=1.5, label=label, alpha=0.8)

        if show_start_end:
            ax.scatter(est_pos[0, 0], est_pos[0, 1], color=color, marker='o', s=60, zorder=10)
            ax.scatter(est_pos[-1, 0], est_pos[-1, 1], color=color, marker='x', s=60, zorder=10)

    # Plot sensor positions
    if sensor_pos is not None:
        ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], color='red', marker='^',
                   s=100, label='Sensor', zorder=15)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')


def plot_error_over_time(
    ax: plt.Axes,
    t: np.ndarray,
    xs_true: np.ndarray,
    estimates: dict,
    pos_indices: tuple = (0, 2),
    colors: Optional[dict] = None,
    linestyles: Optional[dict] = None,
    ylabel: str = 'Position Error',
) -> None:
    """
    Plot position error over time for multiple algorithms.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    t : ndarray [T]
        Time array
    xs_true : ndarray [T, n_x]
        True states
    estimates : dict
        Dictionary mapping algorithm names to estimated states [T, n_x]
    pos_indices : tuple of int
        Indices for x and y position in state vector
    colors : dict, optional
        Color mapping
    linestyles : dict, optional
        Linestyle mapping
    ylabel : str
        Y-axis label
    """
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if colors is None:
        colors = {}
    if linestyles is None:
        linestyles = {}

    ix, iy = pos_indices
    true_pos = np.stack([xs_true[:, ix], xs_true[:, iy]], axis=1)

    for idx, (name, est) in enumerate(estimates.items()):
        est_pos = np.stack([est[:, ix], est[:, iy]], axis=1)
        error = np.sqrt(np.sum((est_pos - true_pos)**2, axis=1))
        color = colors.get(name, default_colors[idx % len(default_colors)])
        ls = linestyles.get(name, '-')

        ax.plot(t, error, color=color, linestyle=ls, linewidth=1.5, label=name, alpha=0.8)

    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
