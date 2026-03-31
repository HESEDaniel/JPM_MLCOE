"""DeepHalo + Sparse Demand Shock Model.

Model:
  u_jt = h_nn(x_jt, S_t) + eta_jt + eps_jt

Two-stage estimation:
  Stage 1: Train DeepHalo NN via MLE (h_nn absorbs beta*X + xi_bar).
  Stage 2: Fix h_nn, MCMC for (eta, gamma, phi).

The NN learns the full deterministic utility including market-level
intercepts (xi_bar), while MCMC recovers sparse demand shocks (eta).

Reference:
    Zhang et al. (2025) for DeepHalo architecture.
    Lu & Shimizu (2025) for sparse demand shock framework.
"""

import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .bayesian_base_model import BayesianChoiceModel
from .deep_halo import FeatureBasedDeepHalo
from .lu25_bayesian import rwmh_step, sample_gamma, sample_phi


def _logit_shares(v):
    """Standard logit shares with outside option. v: (T, J) -> (T, J+1)."""
    T = tf.shape(v)[0]
    v_all = tf.concat([tf.zeros((T, 1)), v], axis=1)
    return tf.nn.softmax(v_all, axis=1)


ShockKernelResults = collections.namedtuple(
    'ShockKernelResults', ['acc_eta'])


class SparseDemandShockKernel(tfp.mcmc.TransitionKernel):
    """Gibbs sweep for (eta, gamma, phi) with fixed h_nn.

    Args:
        h_fixed: (T, J) NN utility (fixed from Stage 1)
        q: (T, J+1) choice counts including outside good
        kappa_eta: RWMH proposal scale for eta
        tau0_sq, tau1_sq: spike and slab variances for eta
        a_phi, b_phi: Beta prior params for phi
    """

    def __init__(self, h_fixed, q, kappa_eta,
                 tau0_sq, tau1_sq, a_phi, b_phi):
        self._h = h_fixed
        self._q = q
        self._kappa_eta = tf.Variable(kappa_eta, trainable=False, dtype=tf.float32)
        self._tau0_sq, self._tau1_sq = tau0_sq, tau1_sq
        self._a_phi, self._b_phi = a_phi, b_phi
        self._Jf = tf.cast(tf.shape(h_fixed)[1], tf.float32)

    @property
    def is_calibrated(self):
        return True

    def one_step(self, state, previous_kernel_results):
        eta, gamma, phi = state
        h, q = self._h, self._q

        # 1. eta (RWMH per market)
        var_eta = gamma * self._tau1_sq + (1.0 - gamma) * self._tau0_sq

        def _rwmh_one_market(args):
            eta_t, h_t, q_t, var_eta_t = args

            def lp_eta(eta_):
                v = h_t + eta_
                v_all = tf.concat([[0.0], v], axis=0)
                shares = tf.clip_by_value(tf.nn.softmax(v_all), 1e-30, 1.0)
                return (tf.reduce_sum(q_t * tf.math.log(shares))
                        - 0.5 * tf.reduce_sum(eta_ ** 2 / var_eta_t))
            new_eta, acc = rwmh_step(eta_t, lp_eta, self._kappa_eta)
            return new_eta, tf.cast(acc, tf.float32)

        new_eta, acc_eta_vec = tf.map_fn(
            _rwmh_one_market, (eta, h, q, var_eta),
            fn_output_signature=(tf.float32, tf.float32))
        eta = new_eta
        acc_eta = tf.reduce_mean(acc_eta_vec)

        # 2. gamma (closed-form Bernoulli)
        gamma = sample_gamma(eta, phi, self._tau0_sq, self._tau1_sq)

        # 3. phi (closed-form Beta)
        phi = sample_phi(gamma, self._a_phi, self._b_phi, self._Jf)

        return (eta, gamma, phi), ShockKernelResults(acc_eta)

    def bootstrap_results(self, init_state):
        return ShockKernelResults(0.0)


class BayesianDeepHalo(BayesianChoiceModel):
    """DeepHalo + Sparse Demand Shock (two-stage estimation).

    Stage 1 (MLE): Train DeepHalo NN (h_nn absorbs beta*X + xi_bar).
    Stage 2 (MCMC): Fix h_nn, sample (eta, gamma, phi).

    Args:
        embedding_dim: dimension of item embeddings
        n_layers: number of DeepHalo layers
        n_heads: number of interaction heads per layer
        nn_lr: learning rate for NN training
        nn_epochs: number of NN training epochs
        tau0_sq: spike variance for eta prior
        tau1_sq: slab variance for eta prior
        a_phi, b_phi: Beta prior parameters for phi
        n_mcmc: total MCMC iterations
        n_burnin: burn-in iterations to discard
    """

    def __init__(self, embedding_dim=32, n_layers=3, n_heads=4,
                 nn_lr=1e-3, nn_epochs=200,
                 tau0_sq=1e-3, tau1_sq=1.0, a_phi=1.0, b_phi=1.0,
                 n_mcmc=5000, n_burnin=1500):
        super().__init__(n_mcmc=n_mcmc, n_burnin=n_burnin, R0=0)
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.nn_lr = nn_lr
        self.nn_epochs = nn_epochs
        self.tau0_sq, self.tau1_sq = tau0_sq, tau1_sq
        self.a_phi, self.b_phi = a_phi, b_phi

    def _train_nn(self, X, q):
        """Stage 1: Train DeepHalo NN via MLE.

        NN learns h_nn which absorbs beta*X + xi_bar (full deterministic utility).
        Stage 2 MCMC then recovers sparse shocks (eta) on top of h_nn.

        Args:
            X: (T, J, d_feat) product features (tf.Tensor)
            q: (T, J+1) choice counts (tf.Tensor)
        """
        T, J = X.shape[0], X.shape[1]
        availability = tf.ones((T, J), dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(self.nn_lr)
        nn = self._nn
        all_vars = nn.trainable_weights

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                h = nn.compute_batch_utility(None, X, availability, None)
                shares = _logit_shares(h)
                nll = -tf.reduce_sum(q * tf.math.log(shares + 1e-10))
            grads = tape.gradient(nll, all_vars)
            optimizer.apply_gradients(
                [(g, w) for g, w in zip(grads, all_vars)
                 if g is not None])
            return nll

        for epoch in range(self.nn_epochs):
            loss = train_step()
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.nn_epochs}: "
                      f"NLL={float(loss):.1f}")

        h_nn = nn.compute_batch_utility(None, X, availability, None)
        print(f"  h_nn range: [{float(tf.reduce_min(h_nn)):.3f}, "
              f"{float(tf.reduce_max(h_nn)):.3f}]")

        self._h_fixed = h_nn

    def _prepare_data(self, X, q):
        self.X = tf.constant(X, dtype=tf.float32)
        self.q = tf.constant(q, dtype=tf.float32)
        self.T, self.J = X.shape[0], X.shape[1]
        self.n_features = X.shape[2]

    def _init_state(self):
        self._init_state_tuple = (
            tf.zeros((self.T, self.J)),       # eta
            tf.zeros((self.T, self.J)),       # gamma
            tf.fill((self.T,), 0.1),          # phi
        )
        self.kappa_eta = 2.38 / float(np.sqrt(self.J))

    def _make_kernel(self):
        return SparseDemandShockKernel(
            self._h_fixed, self.q, self.kappa_eta,
            *[tf.constant(v, tf.float32) for v in [
                self.tau0_sq, self.tau1_sq, self.a_phi, self.b_phi]])

    def _get_kappa_blocks(self, kernel):
        return [
            ("eta", kernel._kappa_eta, "kappa_eta"),
        ]

    def _extract_samples(self, chain, trace):
        eta, gamma, phi = chain
        self.samples_ = {
            "eta": eta.numpy(),
            "gamma": gamma.numpy(),
            "phi": phi.numpy(),
        }
        rate_eta = float(tf.reduce_mean(trace.acc_eta))
        print(f"  Acc: eta={rate_eta:.2f}")

    def fit(self, X, q):
        """Two-stage estimation: NN MLE -> MCMC for sparse shocks.

        Args:
            X: (T, J, d_feat) product features (numpy array)
            q: (T, J+1) choice counts including outside good (numpy array)
        """
        self._prepare_data(X, q)
        self._nn = FeatureBasedDeepHalo(
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
        )
        self._nn.instantiate(self.n_features)

        print("=== Stage 1: Training DeepHalo NN ===")
        self._train_nn(self.X, self.q)

        print("\n=== Stage 2: MCMC for (eta, gamma, phi) ===")
        self._init_state()
        self._calibrate()
        self._run_chain()
        return self

    def refit_shocks(self, X_new, q_new, n_mcmc=None, n_burnin=None):
        """Run Stage 2 MCMC on new markets with fixed NN.

        Args:
            X_new: (T_new, J, d_feat) product features for new markets
            q_new: (T_new, J+1) choice counts for new markets
            n_mcmc: override total MCMC iterations (optional)
            n_burnin: override burn-in iterations (optional)

        Returns:
            (new_shares, new_samples): predicted shares and posterior samples
        """
        orig_X, orig_q, orig_T, orig_J = self.X, self.q, self.T, self.J
        orig_h_fixed = self._h_fixed
        orig_samples, orig_fitted = self.samples_, self.is_fitted
        orig_n_mcmc, orig_n_burnin = self.n_mcmc, self.n_burnin

        self._prepare_data(X_new, q_new)
        T_new, J_new = X_new.shape[0], X_new.shape[1]
        availability = tf.ones((T_new, J_new), dtype=tf.float32)
        h_nn = self._nn.compute_batch_utility(
            None, tf.constant(X_new, tf.float32), availability, None)
        self._h_fixed = h_nn

        if n_mcmc is not None:
            self.n_mcmc = n_mcmc
        if n_burnin is not None:
            self.n_burnin = n_burnin

        self._init_state()
        self._calibrate()
        self._run_chain()
        new_samples = self.samples_
        new_full_chain = self.full_chain_

        eta = tf.constant(np.mean(new_samples["eta"], 0), tf.float32)
        v = self._h_fixed + eta
        new_shares = _logit_shares(v).numpy()

        self.X, self.q, self.T, self.J = orig_X, orig_q, orig_T, orig_J
        self._h_fixed = orig_h_fixed
        self.samples_, self.is_fitted = orig_samples, orig_fitted
        self.n_mcmc, self.n_burnin = orig_n_mcmc, orig_n_burnin

        return new_shares, new_samples, new_full_chain

    def compute_shares(self, X):
        """Predict shares using h_nn + posterior mean eta.

        Args:
            X: unused (h_fixed already computed), kept for API compatibility
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        eta = tf.constant(np.mean(self.samples_["eta"], 0), tf.float32)
        v = self._h_fixed + eta
        return _logit_shares(v).numpy()

    def get_posterior_summary(self):
        """Posterior summary for eta."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        def _s(x):
            return {"mean": np.mean(x, 0), "std": np.std(x, 0),
                    "ci_lower": np.percentile(x, 2.5, 0),
                    "ci_upper": np.percentile(x, 97.5, 0)}

        return {"eta": _s(self.samples_["eta"])}

    def get_sparsity_summary(self):
        """Sparsity pattern summary from posterior gamma samples."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        gamma = self.samples_["gamma"]
        return {
            "prob_gamma_1": np.mean(gamma, axis=0),
            "frac_active": np.mean(np.mean(gamma, axis=0) > 0.5),
        }
