"""
Visualization utilities for particle filters and state estimation.

This module provides plotting functions organized by domain:
- filters: Kalman filter family visualizations
- particles: Particle filter specific plots (OMAT, ESS, boxplots, weight distributions)
- tracking: Trajectory and spatial plots (ellipses, contours, trajectory comparison)
- tables: Metrics tables and summaries
"""
from .filters import (
    plot_kalman_filter,
    plot_stability_analysis,
    plot_filter_estimate_with_bands,
    plot_multi_filter_comparison,
    plot_filter_comparison,
    plot_linearization_analysis,
    plot_sigma_point_analysis,
    plot_nees_comparison,
    plot_model_comparison_side_by_side,
)

from .particles import (
    plot_omat_comparison,
    plot_ess_comparison,
    plot_omat_boxplot,
    plot_weight_distributions,
    DEFAULT_COLORS,
    DEFAULT_LINESTYLES,
)

from .tracking import (
    plot_covariance_ellipse,
    plot_contours,
    plot_trajectory_2d,
    plot_particle_cloud,
    plot_trajectory_comparison,
    plot_error_over_time,
)

from .tables import (
    save_metrics_table,
    format_runtime,
    metrics_to_latex,
)

__all__ = [
    # filters
    'plot_kalman_filter',
    'plot_stability_analysis',
    'plot_filter_estimate_with_bands',
    'plot_multi_filter_comparison',
    'plot_filter_comparison',
    'plot_linearization_analysis',
    'plot_sigma_point_analysis',
    'plot_nees_comparison',
    'plot_model_comparison_side_by_side',
    # particles
    'plot_omat_comparison',
    'plot_ess_comparison',
    'plot_omat_boxplot',
    'plot_weight_distributions',
    'DEFAULT_COLORS',
    'DEFAULT_LINESTYLES',
    # tracking
    'plot_covariance_ellipse',
    'plot_contours',
    'plot_trajectory_2d',
    'plot_particle_cloud',
    'plot_trajectory_comparison',
    'plot_error_over_time',
    # tables
    'save_metrics_table',
    'format_runtime',
    'metrics_to_latex',
]
