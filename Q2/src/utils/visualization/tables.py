"""
Utilities for generating metrics tables and summaries.
"""
from typing import Dict, List, Optional, Union


def save_metrics_table(
    metrics: Dict[str, Dict],
    save_path: str,
    columns: List[str],
    title: str = 'Performance Comparison',
    column_widths: Optional[Dict[str, int]] = None,
    float_format: str = '.3f',
    na_string: str = 'N/A'
) -> None:
    """
    Save metrics table to a text file.

    Parameters
    ----------
    metrics : dict
        Dictionary mapping algorithm names to metric dictionaries.
        Each metric dict contains column_name -> value pairs.
    save_path : str
        Path to save the table
    columns : list of str
        Column names to include (in order)
    title : str
        Table title
    column_widths : dict, optional
        Column name -> width mapping. If None, uses default widths.
    float_format : str
        Format string for float values
    na_string : str
        String to display for missing values

    Example
    -------
    >>> metrics = {
    ...     'EKF': {'mean_omat': 1.234, 'std_omat': 0.123, 'mean_runtime': 0.05},
    ...     'UKF': {'mean_omat': 1.456, 'std_omat': 0.234, 'mean_runtime': 0.08},
    ... }
    >>> save_metrics_table(
    ...     metrics, 'results.txt',
    ...     columns=['mean_omat', 'std_omat', 'mean_runtime'],
    ...     title='Filter Performance'
    ... )
    """
    # Default column widths
    if column_widths is None:
        column_widths = {col: 12 for col in columns}
        column_widths['Algorithm'] = 15

    # Build header
    algo_width = column_widths.get('Algorithm', 15)
    header_parts = [f"{'Algorithm':<{algo_width}}"]
    for col in columns:
        width = column_widths.get(col, 12)
        # Format column name for display
        display_name = col.replace('_', ' ').title()
        header_parts.append(f"{display_name:>{width}}")
    header = ' '.join(header_parts)

    # Calculate total width
    total_width = len(header)

    with open(save_path, 'w') as f:
        f.write('=' * total_width + '\n')
        f.write(f'{title}\n')
        f.write('=' * total_width + '\n\n')
        f.write(header + '\n')
        f.write('-' * total_width + '\n')

        for algo, m in metrics.items():
            row_parts = [f"{algo:<{algo_width}}"]
            for col in columns:
                width = column_widths.get(col, 12)
                value = m.get(col)
                if value is None:
                    formatted = na_string
                elif isinstance(value, float):
                    formatted = f"{value:{float_format}}"
                else:
                    formatted = str(value)
                row_parts.append(f"{formatted:>{width}}")
            f.write(' '.join(row_parts) + '\n')

        f.write('=' * total_width + '\n')

    print(f'Table saved to: {save_path}')


def format_runtime(seconds: float) -> str:
    """
    Format runtime in human-readable form.

    Parameters
    ----------
    seconds : float
        Runtime in seconds

    Returns
    -------
    str
        Formatted runtime string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.1f}us"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def metrics_to_latex(
    metrics: Dict[str, Dict],
    columns: List[str],
    caption: str = 'Performance Comparison',
    label: str = 'tab:performance',
    float_format: str = '.3f'
) -> str:
    """
    Convert metrics dictionary to LaTeX table string.

    Parameters
    ----------
    metrics : dict
        Dictionary mapping algorithm names to metric dictionaries
    columns : list of str
        Column names to include
    caption : str
        Table caption
    label : str
        LaTeX label
    float_format : str
        Format string for float values

    Returns
    -------
    str
        LaTeX table code
    """
    n_cols = len(columns) + 1  # +1 for algorithm name

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\begin{tabular}{l' + 'r' * len(columns) + '}',
        r'\toprule',
    ]

    # Header row
    col_names = ['Algorithm'] + [col.replace('_', ' ').title() for col in columns]
    lines.append(' & '.join(col_names) + r' \\')
    lines.append(r'\midrule')

    # Data rows
    for algo, m in metrics.items():
        row = [algo]
        for col in columns:
            value = m.get(col)
            if value is None:
                row.append('--')
            elif isinstance(value, float):
                row.append(f"{value:{float_format}}")
            else:
                row.append(str(value))
        lines.append(' & '.join(row) + r' \\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)
