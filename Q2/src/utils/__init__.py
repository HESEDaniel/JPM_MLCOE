"""
Utility Functions.

This module contains utility functions for:
- Metrics computation
- Visualization (organized in visualization/ subfolder)
- Experiment logging
- Numerical utilities for particle flows
"""
from .metrics import compute_mse, stability_summary
from .experiment_logger import ExperimentLogger
from .utils import exponential_lambda_schedule

# Visualization imports from subfolder
from .visualization import (
    # filters
    plot_kalman_filter,
    plot_stability_analysis,
    # particles
    plot_omat_comparison,
    plot_ess_comparison,
    plot_omat_boxplot,
    # tracking
    plot_covariance_ellipse,
    plot_contours,
    plot_trajectory_2d,
    plot_particle_cloud,
    # tables
    save_metrics_table,
    format_runtime,
    metrics_to_latex,
)

__all__ = [
    # metrics
    'compute_mse',
    'stability_summary',
    # experiment logger
    'ExperimentLogger',
    # utils
    'exponential_lambda_schedule',
    # visualization - filters
    'plot_kalman_filter',
    'plot_stability_analysis',
    # visualization - particles
    'plot_omat_comparison',
    'plot_ess_comparison',
    'plot_omat_boxplot',
    # visualization - tracking
    'plot_covariance_ellipse',
    'plot_contours',
    'plot_trajectory_2d',
    'plot_particle_cloud',
    # visualization - tables
    'save_metrics_table',
    'format_runtime',
    'metrics_to_latex',
]
