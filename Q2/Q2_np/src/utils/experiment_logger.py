"""
Experiment Logger - Track experiment configurations and results.

Provides per-algorithm caching with numpy arrays saved to disk.
Results are cached individually per algorithm, allowing mixed cached/fresh runs.
"""
import os
import csv
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union


class ExperimentLogger:
    """
    Logger for tracking experiment configurations and per-algorithm results.

    Features:
    - Per-algorithm result caching (numpy arrays saved as .npz files)
    - Single CSV log file tracking individual algorithm runs
    - Timestamped run directories for plots and reports

    Usage:
        logger = ExperimentLogger(experiment_name='exp_2_1a_li17')

        # Define experiment config (excludes algorithm list)
        config = {'T': 40, 'N_particles': 500, 'seed': 42, ...}

        # Check if algorithm result is cached
        if logger.algorithm_result_exists('EKF', **config):
            result = logger.load_algorithm_result('EKF', **config)
        else:
            result = run_ekf(...)
            logger.save_algorithm_result('EKF', config, result)

        # Create timestamped directory for plots
        run_dir = logger.create_timestamped_run_dir()
    """

    # Columns for per-algorithm log
    LOG_COLUMNS = [
        'timestamp', 'experiment_name', 'algorithm',
        'T', 'N_particles', 'n_flow_steps', 'n_trajectories', 'n_trials', 'seed',
        'mean_omat', 'std_omat', 'mean_ess', 'mean_resamples', 'runtime_sec',
        'cache_file', 'status', 'notes'
    ]

    # Config keys used for cache matching (algorithm-specific)
    CACHE_KEYS = [
        'T', 'N_particles', 'n_flow_steps', 'n_trajectories', 'n_trials', 'seed'
    ]

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        results_root: Optional[str] = None
    ):
        """
        Initialize experiment logger.

        Parameters
        ----------
        experiment_name : str, optional
            Name of the experiment (e.g., 'exp_2_1a_li17').
            If provided, logs are stored in results/{experiment_name}/.
        log_dir : str, optional
            [Deprecated] Directory to store log files. Use experiment_name instead.
        results_root : str, optional
            Root directory for results (default: Q2/results).
        """
        # Determine Q2 root directory
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self._q2_root = os.path.normpath(os.path.join(self._script_dir, '..', '..'))

        # Set results root
        if results_root is not None:
            self._results_root = results_root
        else:
            self._results_root = os.path.join(self._q2_root, 'results')

        self.experiment_name = experiment_name

        # Determine log directory
        if experiment_name is not None:
            self.log_dir = os.path.join(self._results_root, experiment_name)
        elif log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = self._results_root

        # Single log file for all algorithm runs
        self.log_file = os.path.join(self.log_dir, "algorithm_log.csv")

        # Cache directory for numpy result files
        self.cache_dir = os.path.join(self.log_dir, "cache")

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize CSV file with headers if it doesn't exist
        self._init_csv_file(self.log_file, self.LOG_COLUMNS)

        # Track current run timestamp (set by create_timestamped_run_dir)
        self._current_timestamp: Optional[str] = None
        self._current_run_dir: Optional[str] = None

    def _init_csv_file(self, filepath: str, columns: List[str]) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

    def _get_config_hash(self, algorithm: str, config: Dict) -> str:
        """
        Generate a hash string for the algorithm + config combination.

        Parameters
        ----------
        algorithm : str
            Algorithm name (e.g., 'EKF', 'PF-PF(EDH)')
        config : dict
            Experiment configuration

        Returns
        -------
        str
            Hash string for cache filename
        """
        # Build key string from relevant config values
        key_parts = [algorithm]
        for k in self.CACHE_KEYS:
            if k in config:
                key_parts.append(f"{k}={config[k]}")

        key_str = "_".join(key_parts)
        # Create short hash for filename
        hash_str = hashlib.md5(key_str.encode()).hexdigest()[:12]
        return hash_str

    def _get_cache_filename(self, algorithm: str, config: Dict) -> str:
        """
        Get the cache filename for an algorithm + config.

        Parameters
        ----------
        algorithm : str
            Algorithm name
        config : dict
            Experiment configuration

        Returns
        -------
        str
            Filename (without path) for the cache file
        """
        # Sanitize algorithm name for filename
        safe_algo = algorithm.replace('(', '_').replace(')', '').replace(' ', '_')
        config_hash = self._get_config_hash(algorithm, config)
        return f"{safe_algo}_{config_hash}.npz"

    def get_cache_path(self, algorithm: str, config: Dict) -> str:
        """
        Get the full path to the cache file for an algorithm.

        Parameters
        ----------
        algorithm : str
            Algorithm name
        config : dict
            Experiment configuration

        Returns
        -------
        str
            Full path to cache file
        """
        filename = self._get_cache_filename(algorithm, config)
        return os.path.join(self.cache_dir, filename)

    def algorithm_result_exists(self, algorithm: str, **config) -> bool:
        """
        Check if a cached result exists for the algorithm + config.

        Parameters
        ----------
        algorithm : str
            Algorithm name
        **config : dict
            Experiment configuration

        Returns
        -------
        bool
            True if cached result exists
        """
        cache_path = self.get_cache_path(algorithm, config)
        return os.path.exists(cache_path)

    def save_algorithm_result(
        self,
        algorithm: str,
        config: Dict[str, Any],
        data: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        runtime_sec: float = 0.0,
        notes: str = ''
    ) -> str:
        """
        Save algorithm result to cache.

        Parameters
        ----------
        algorithm : str
            Algorithm name
        config : dict
            Experiment configuration
        data : dict
            Result data to cache. Values can be numpy arrays or scalars.
            Example: {'m_filt': array, 'P_filt': array, 'omat': array, 'ess': array}
        metrics : dict, optional
            Summary metrics {mean_omat, std_omat, mean_ess, mean_resamples}
        runtime_sec : float
            Algorithm runtime in seconds
        notes : str
            Optional notes

        Returns
        -------
        str
            Path to saved cache file
        """
        cache_path = self.get_cache_path(algorithm, config)

        # Save numpy arrays
        np.savez_compressed(cache_path, **data)

        # Log to CSV
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        row = {
            'timestamp': timestamp,
            'experiment_name': self.experiment_name or '',
            'algorithm': algorithm,
            'T': config.get('T', ''),
            'N_particles': config.get('N_particles', ''),
            'n_flow_steps': config.get('n_flow_steps', ''),
            'n_trajectories': config.get('n_trajectories', ''),
            'n_trials': config.get('n_trials', ''),
            'seed': config.get('seed', ''),
            'mean_omat': f"{metrics.get('mean_omat', ''):.4f}" if metrics and metrics.get('mean_omat') is not None else '',
            'std_omat': f"{metrics.get('std_omat', ''):.4f}" if metrics and metrics.get('std_omat') is not None else '',
            'mean_ess': f"{metrics.get('mean_ess', ''):.2f}" if metrics and metrics.get('mean_ess') is not None else '',
            'mean_resamples': f"{metrics.get('mean_resamples', ''):.1f}" if metrics and metrics.get('mean_resamples') is not None else '',
            'runtime_sec': f"{runtime_sec:.2f}",
            'cache_file': os.path.basename(cache_path),
            'status': 'completed',
            'notes': notes,
        }

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.LOG_COLUMNS)
            writer.writerow(row)

        print(f"  Cached {algorithm}: {os.path.basename(cache_path)}")
        return cache_path

    def load_algorithm_result(self, algorithm: str, **config) -> Optional[Dict[str, np.ndarray]]:
        """
        Load cached algorithm result.

        Parameters
        ----------
        algorithm : str
            Algorithm name
        **config : dict
            Experiment configuration

        Returns
        -------
        dict or None
            Dictionary of numpy arrays if cache exists, else None.
            Keys depend on what was saved (e.g., m_filt, P_filt, omat, ess)
        """
        cache_path = self.get_cache_path(algorithm, config)

        if not os.path.exists(cache_path):
            return None

        # Load numpy arrays
        with np.load(cache_path, allow_pickle=True) as npz:
            data = {key: npz[key] for key in npz.files}

        print(f"  Loaded cached {algorithm}: {os.path.basename(cache_path)}")
        return data

    def get_cached_algorithms(self, **config) -> List[str]:
        """
        Get list of algorithms that have cached results for the given config.

        Parameters
        ----------
        **config : dict
            Experiment configuration

        Returns
        -------
        list[str]
            List of algorithm names with cached results
        """
        cached = []
        if not os.path.exists(self.log_file):
            return cached

        with open(self.log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') != 'completed':
                    continue

                # Check if config matches
                matches = True
                for key in self.CACHE_KEYS:
                    if key in config:
                        if str(row.get(key, '')) != str(config[key]):
                            matches = False
                            break

                if matches:
                    algo = row['algorithm']
                    # Verify cache file exists
                    cache_file = row.get('cache_file', '')
                    if cache_file and os.path.exists(os.path.join(self.cache_dir, cache_file)):
                        if algo not in cached:
                            cached.append(algo)

        return cached

    def create_timestamped_run_dir(self, timestamp: Optional[str] = None) -> str:
        """
        Create a timestamped directory for plots and reports.

        Parameters
        ----------
        timestamp : str, optional
            Custom timestamp string. If None, uses current time.
            Format: YYYY-MM-DD_HH-MM-SS

        Returns
        -------
        str
            Path to the created timestamped run directory.
        """
        if timestamp is None:
            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')

        self._current_timestamp = timestamp
        self._current_run_dir = os.path.join(self.log_dir, timestamp)

        os.makedirs(self._current_run_dir, exist_ok=True)

        return self._current_run_dir

    def get_run_dir(self) -> Optional[str]:
        """Get the current run directory."""
        return self._current_run_dir

    def get_figures_dir(self, create: bool = True) -> str:
        """Get the figures directory for the current run."""
        if self._current_run_dir is None:
            raise RuntimeError("Call create_timestamped_run_dir() first")

        figs_dir = os.path.join(self._current_run_dir, 'figures')
        if create:
            os.makedirs(figs_dir, exist_ok=True)
        return figs_dir

    def get_metrics_dir(self, create: bool = True) -> str:
        """Get the metrics directory for the current run."""
        if self._current_run_dir is None:
            raise RuntimeError("Call create_timestamped_run_dir() first")

        metrics_dir = os.path.join(self._current_run_dir, 'metrics')
        if create:
            os.makedirs(metrics_dir, exist_ok=True)
        return metrics_dir

    def print_cache_summary(self, **config) -> None:
        """Print summary of cached algorithm results."""
        cached = self.get_cached_algorithms(**config)

        print("Cached Algorithm Results")

        if not cached:
            print("No cached results found for this configuration.")
        else:
            print(f"Found {len(cached)} cached algorithms:")
            for algo in cached:
                cache_path = self.get_cache_path(algo, config)
                size_kb = os.path.getsize(cache_path) / 1024
                print(f"  - {algo}: {os.path.basename(cache_path)} ({size_kb:.1f} KB)")

    def clear_algorithm_cache(self, algorithm: str, **config) -> bool:
        """
        Remove cached result for a specific algorithm.

        Parameters
        ----------
        algorithm : str
            Algorithm name
        **config : dict
            Experiment configuration

        Returns
        -------
        bool
            True if cache was removed, False if it didn't exist
        """
        cache_path = self.get_cache_path(algorithm, config)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Removed cache: {os.path.basename(cache_path)}")
            return True
        return False

    def clear_all_cache(self) -> int:
        """
        Remove all cached results for this experiment.

        Returns
        -------
        int
            Number of cache files removed
        """
        count = 0
        if os.path.exists(self.cache_dir):
            for f in os.listdir(self.cache_dir):
                if f.endswith('.npz'):
                    os.remove(os.path.join(self.cache_dir, f))
                    count += 1
        print(f"Removed {count} cache files.")
        return count

    def log_experiment(self, config: Dict[str, Any], duration_sec: float = 0.0,
                       notes: str = '') -> None:
        """
        Log experiment completion.

        Parameters
        ----------
        config : dict
            Experiment configuration
        duration_sec : float
            Total duration in seconds
        notes : str
            Optional notes
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_path = os.path.join(self.log_dir, 'experiment_log.txt')

        with open(log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Duration: {duration_sec:.1f}s\n")
            for key, val in config.items():
                f.write(f"  {key}: {val}\n")
            if notes:
                f.write(f"Notes: {notes}\n")

        print(f"Experiment completed in {duration_sec:.1f}s")
