"""Base class for Bayesian MCMC choice models.

Provides common infrastructure: adaptive kappa calibration, chain execution
via tfp.mcmc.sample_chain, and posterior summary statistics.

Subclasses must implement:
  _prepare_data, _init_state, _make_kernel, _get_kappa_blocks,
  _extract_samples, compute_shares
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class BayesianChoiceModel:
    """Base class for Bayesian MCMC choice models.

    Args:
        n_mcmc: total MCMC iterations
        n_burnin: burn-in iterations to discard
        R0: number of MC draws for share approximation
    """

    def __init__(self, n_mcmc=5000, n_burnin=1500, R0=200):
        self.n_mcmc = n_mcmc
        self.n_burnin = n_burnin
        self.R0 = R0
        self.is_fitted = False
        self.samples_ = None

    def _prepare_data(self, X, q):
        """Convert input data to TF tensors and set dimension attributes.

        Args:
            X: (T, J, d_X) product characteristics
            q: (T, J+1) choice counts including outside option
        """
        raise NotImplementedError

    def _init_state(self):
        """Set initial MCMC state. Must set self._init_state_tuple."""
        raise NotImplementedError

    def _make_kernel(self):
        """Create and return a tfp.mcmc.TransitionKernel."""
        raise NotImplementedError

    def _get_kappa_blocks(self, kernel):
        """Return list of (name, tf.Variable, attr_name) for calibration.

        Args:
            kernel: the TransitionKernel created by _make_kernel()
        """
        raise NotImplementedError

    def _extract_samples(self, chain, trace):
        """Extract posterior samples from chain output into self.samples_.

        Args:
            chain: tuple of tensors from sample_chain
            trace: kernel results trace
        """
        raise NotImplementedError

    def compute_shares(self, X):
        """Predict market shares using posterior means.

        Args:
            X: (T, J, d_X) product characteristics
        """
        raise NotImplementedError

    def _calibrate(self, n_calib=20, max_rounds=30,
                   target_low=0.3, target_high=0.5):
        """Adaptive kappa calibration via pilot runs.

        Args:
            n_calib: iterations per pilot run
            max_rounds: max calibration rounds
            target_low: lower bound for acceptance rate
            target_high: upper bound for acceptance rate
        """
        kernel = self._make_kernel()
        blocks = self._get_kappa_blocks(kernel)

        @tf.function
        def _pilot(state):
            return tfp.mcmc.sample_chain(
                num_results=n_calib, current_state=state, kernel=kernel,
                trace_fn=lambda _, kr: kr, return_final_kernel_results=True)

        for _ in range(max_rounds):
            chain, trace, _ = _pilot(self._init_state_tuple)
            self._init_state_tuple = tuple(s[-1] for s in chain)

            rates = {name: float(tf.reduce_mean(getattr(trace, f'acc_{name}')))
                     for name, _, _ in blocks}

            all_ok = True
            for name, kappa_var, attr_name in blocks:
                if rates[name] < target_low:
                    kappa_var.assign(kappa_var * 0.7)
                    setattr(self, attr_name, float(kappa_var))
                    all_ok = False
                elif rates[name] > target_high:
                    kappa_var.assign(kappa_var * 1.3)
                    setattr(self, attr_name, float(kappa_var))
                    all_ok = False

            if all_ok:
                break

        self._kernel = kernel

    def _run_chain(self):
        """Run the main MCMC chain, storing full chain for diagnostics."""
        kernel = self._kernel

        @tf.function
        def _run(state):
            return tfp.mcmc.sample_chain(
                num_results=self.n_mcmc, num_burnin_steps=0,
                current_state=state, kernel=kernel,
                trace_fn=lambda _, kr: kr, return_final_kernel_results=True)

        full_chain, full_trace, _ = _run(self._init_state_tuple)

        # Store full chain for convergence diagnostics
        self.full_chain_ = tuple(s.numpy() for s in full_chain)
        self.full_trace_ = full_trace

        # Pass burn-in trimmed chain to _extract_samples
        chain = tuple(s[self.n_burnin:] for s in full_chain)
        self._extract_samples(chain, full_trace)
        self.is_fitted = True

    def fit(self, X, q):
        """Full estimation: prepare data -> init -> calibrate -> run chain.

        Args:
            X: (T, J, d_X) product characteristics
            q: (T, J+1) choice counts including outside option
        """
        self._prepare_data(X, q)
        self._init_state()
        self._calibrate()
        self._run_chain()
        return self

    def predict_probas(self, X):
        """Predict choice probabilities.

        Args:
            X: (T, J, d_X) product characteristics
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.compute_shares(X)

    def get_posterior_summary(self):
        """Return posterior summary: mean, std, 95% credible interval."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        def _summarize(x):
            return {"mean": np.mean(x, 0), "std": np.std(x, 0),
                    "ci_lower": np.percentile(x, 2.5, 0),
                    "ci_upper": np.percentile(x, 97.5, 0)}

        return {k: _summarize(v) for k, v in self.samples_.items()}
