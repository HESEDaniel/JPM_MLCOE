"""Lu & Shimizu (2025) Sparse Demand Shock Model.

MCMC algorithm:
  1. beta_bar (TMH)  -- mean taste parameters
  2. r       (RWMH)  -- log standard deviations of random coefficients
  3. xi_bar  (RWMH)  -- market-specific intercepts (batched over T)
  4. eta     (TMH)   -- market-product shocks (per market via map_fn)
  5. gamma   (closed-form Bernoulli) -- sparsity indicators
  6. phi     (closed-form Beta)      -- inclusion probabilities
"""

import collections

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from choice_learn.datasets.dgp import simulate_lu25_dgp  # noqa: F401

from .bayesian_base_model import BayesianChoiceModel

tfd = tfp.distributions


def sample_gamma(eta, phi, tau0_sq, tau1_sq):
    """Sample sparsity indicators from closed-form Bernoulli.

    Args:
        eta: (T, J) current demand shocks
        phi: (T,) current inclusion probabilities
        tau0_sq: spike variance (scalar)
        tau1_sq: slab variance (scalar)

    Returns:
        (T, J) sampled gamma (0=sparse, 1=non-sparse)
    """
    spike = tfd.Normal(0.0, tf.sqrt(tau0_sq))
    slab = tfd.Normal(0.0, tf.sqrt(tau1_sq))
    log_p1 = (tf.math.log(tf.clip_by_value(phi[:, tf.newaxis], 1e-30, 1.0))
              + slab.log_prob(eta))
    log_p0 = (tf.math.log(tf.clip_by_value(1.0 - phi[:, tf.newaxis], 1e-30, 1.0))
              + spike.log_prob(eta))
    return tfd.Bernoulli(
        probs=tf.sigmoid(tf.clip_by_value(log_p1 - log_p0, -500., 500.)),
        dtype=tf.float32).sample()


def sample_phi(gamma, a_phi, b_phi, J):
    """Sample inclusion probabilities from closed-form Beta.

    Args:
        gamma: (T, J) sparsity indicators
        a_phi: Beta prior parameter
        b_phi: Beta prior parameter
        J: number of products (scalar or tf scalar)

    Returns:
        (T,) sampled phi
    """
    n1 = tf.reduce_sum(gamma, axis=1)
    return tfd.Beta(a_phi + n1, b_phi + J - n1).sample()


def rwmh_step(current, log_posterior_fn, kappa):
    """Random Walk Metropolis-Hastings step.

    Args:
        current: current parameter value, (D,) or (T,)
        log_posterior_fn: callable returning log-posterior, same shape as current
        kappa: proposal scale (scalar)

    Returns:
        (new_value, accepted): updated parameter and boolean acceptance mask
    """
    proposal = current + kappa * tf.random.normal(tf.shape(current))
    log_alpha = tf.minimum(log_posterior_fn(proposal) - log_posterior_fn(current), 0.0)
    accepted = tf.math.log(tf.random.uniform(tf.shape(log_alpha))) < log_alpha
    return tf.where(accepted, proposal, current), accepted


def tmh_step(current, log_posterior_fn, kappa, n_newton=2):
    """Tailored Metropolis-Hastings step (Paper Appendix B.1).

    Newton mode-finding -> Hessian-based proposal -> asymmetric correction.

    Args:
        current: current parameter value (D,)
        log_posterior_fn: callable theta -> scalar log-posterior
        kappa: proposal scale
        n_newton: number of Newton iterations

    Returns:
        (new_value, accepted): updated parameter and boolean acceptance flag
    """
    d = tf.shape(current)[0]
    theta = tf.identity(current)

    for _ in range(n_newton):
        with tf.GradientTape() as t2:
            t2.watch(theta)
            with tf.GradientTape() as t1:
                t1.watch(theta)
                lp = log_posterior_fn(theta)
            g = t1.gradient(lp, theta)
        H = t2.jacobian(g, theta)
        step = tf.linalg.solve(H[tf.newaxis], g[tf.newaxis, :, tf.newaxis])[0, :, 0]
        theta_new = theta - step
        theta = tf.cond(tf.reduce_all(tf.math.is_finite(theta_new)),
                        lambda: theta_new, lambda: theta)

    with tf.GradientTape() as t2:
        t2.watch(theta)
        with tf.GradientTape() as t1:
            t1.watch(theta)
            lp = log_posterior_fn(theta)
        g = t1.gradient(lp, theta)
    H_mode = t2.jacobian(g, theta)

    V_hat = -tf.linalg.inv(H_mode)
    V_hat = 0.5 * (V_hat + tf.transpose(V_hat))
    L = tf.linalg.cholesky(V_hat)

    z = tf.random.normal((d,))
    proposal = theta + kappa * tf.linalg.matvec(L, z)

    ok = tf.reduce_all(tf.math.is_finite(proposal))

    lp_curr = log_posterior_fn(current)
    lp_prop = log_posterior_fn(proposal)
    V_inv = -H_mode / (kappa ** 2)
    log_q_curr = -0.5 * tf.einsum('i,ij,j->', current - theta, V_inv, current - theta)
    log_q_prop = -0.5 * tf.einsum('i,ij,j->', proposal - theta, V_inv, proposal - theta)
    log_alpha = tf.minimum((lp_prop + log_q_curr) - (lp_curr + log_q_prop), 0.0)

    accepted = ok & (tf.math.log(tf.random.uniform(())) < log_alpha)
    return tf.cond(accepted, lambda: proposal, lambda: current), accepted


def _shares_all(beta_bar, r, xi_bar, eta, X, v_draws, rc_indices, h_offset=None):
    """Predicted shares for all markets.

    Args:
        beta_bar: (d_X,) mean taste parameters
        r: (d_rc,) log standard deviations
        xi_bar: (T,) market intercepts
        eta: (T, J) market-product shocks
        X: (T, J, d_X) product characteristics
        v_draws: (R0, d_rc) MC draws
        rc_indices: indices of covariates with random coefficients
        h_offset: optional (T, J) context effect

    Returns:
        (T, J+1) predicted shares
    """
    T, R0 = tf.shape(X)[0], tf.shape(v_draws)[0]
    delta = tf.reduce_sum(X * beta_bar[tf.newaxis, tf.newaxis, :], axis=-1)
    delta = delta + xi_bar[:, tf.newaxis] + eta
    if h_offset is not None:
        delta = delta + h_offset
    X_rc = tf.gather(X, rc_indices, axis=2)
    mu = tf.einsum("tjk,rk->tjr",
                   X_rc * tf.exp(r)[tf.newaxis, tf.newaxis, :], v_draws)
    util = tf.concat([tf.zeros((T, 1, R0)),
                      delta[:, :, tf.newaxis] + mu], axis=1)
    max_u = tf.reduce_max(util, axis=1, keepdims=True)
    exp_u = tf.exp(util - max_u)
    s = tf.reduce_mean(exp_u / tf.reduce_sum(exp_u, axis=1, keepdims=True),
                       axis=2)
    return tf.clip_by_value(s, 1e-30, 1.0)


def _shares_single(beta_bar, r, xi_bar_t, eta_t, X_t, v_draws, rc_indices,
                   h_offset_t=None):
    """Predicted shares for one market.

    Args:
        beta_bar: (d_X,) mean taste parameters
        r: (d_rc,) log standard deviations
        xi_bar_t: scalar, market intercept
        eta_t: (J,) market-product shocks
        X_t: (J, d_X) product characteristics for one market
        v_draws: (R0, d_rc) MC draws
        rc_indices: indices of covariates with random coefficients
        h_offset_t: optional (J,) context effect

    Returns:
        (J+1,) predicted shares
    """
    delta = tf.reduce_sum(X_t * beta_bar[tf.newaxis, :], axis=-1) + xi_bar_t + eta_t
    if h_offset_t is not None:
        delta = delta + h_offset_t
    X_rc = tf.gather(X_t, rc_indices, axis=1)
    R0 = tf.shape(v_draws)[0]
    mu = tf.einsum("jk,rk->jr", X_rc * tf.exp(r)[tf.newaxis, :], v_draws)
    util = tf.concat([tf.zeros((1, R0)), delta[:, tf.newaxis] + mu], axis=0)
    max_u = tf.reduce_max(util, axis=0, keepdims=True)
    exp_u = tf.exp(util - max_u)
    s = tf.reduce_mean(exp_u / tf.reduce_sum(exp_u, axis=0, keepdims=True), axis=1)
    return tf.clip_by_value(s, 1e-30, 1.0)


def _ll_per_market(beta_bar, r, xi_bar, eta, X, q, v_draws, rc_indices,
                   h_offset=None):
    """Per-market log-likelihood. Returns (T,)."""
    shares = _shares_all(beta_bar, r, xi_bar, eta, X, v_draws, rc_indices,
                         h_offset)
    return tf.reduce_sum(q * tf.math.log(shares), axis=1)


def _ll_total(beta_bar, r, xi_bar, eta, X, q, v_draws, rc_indices,
              h_offset=None):
    """Total log-likelihood. Returns scalar."""
    return tf.reduce_sum(
        _ll_per_market(beta_bar, r, xi_bar, eta, X, q, v_draws, rc_indices,
                       h_offset))


Lu25KernelResults = collections.namedtuple(
    'Lu25KernelResults', ['acc_beta', 'acc_r', 'acc_xi', 'acc_eta'])


class Lu25MCMCKernel(tfp.mcmc.TransitionKernel):
    """Full sweep: beta(TMH) -> r(RWMH) -> xi(RWMH) -> eta(TMH) -> gamma -> phi.

    Args:
        X: (T, J, d_X) product characteristics
        q: (T, J+1) choice counts
        v_draws: (R0, d_rc) MC draws
        rc_indices: indices of random coefficient covariates
        kappa_beta, kappa_r, kappa_xi, kappa_eta: proposal scales
        mu_beta, V_beta: prior for beta_bar
        mu_xi, V_xi: prior for xi_bar
        V_r: prior variance for r
        tau0_sq, tau1_sq: spike and slab variances
        a_phi, b_phi: Beta prior for phi
        h_offset: optional (T, J) context effect
    """

    def __init__(self, X, q, v_draws, rc_indices,
                 kappa_beta, kappa_r, kappa_xi, kappa_eta,
                 mu_beta, V_beta, mu_xi, V_xi, V_r,
                 tau0_sq, tau1_sq, a_phi, b_phi,
                 h_offset=None, use_tmh=True):
        self._X, self._q, self._v_draws, self._rc_indices = X, q, v_draws, rc_indices
        if h_offset is not None:
            self._h_offset = tf.Variable(h_offset, trainable=False, dtype=tf.float32)
        else:
            self._h_offset = None
        self._kappa_beta = tf.Variable(kappa_beta, trainable=False, dtype=tf.float32)
        self._kappa_r = tf.Variable(kappa_r, trainable=False, dtype=tf.float32)
        self._kappa_xi = tf.Variable(kappa_xi, trainable=False, dtype=tf.float32)
        self._kappa_eta = tf.Variable(kappa_eta, trainable=False, dtype=tf.float32)
        self._mu_beta, self._V_beta = mu_beta, V_beta
        self._mu_xi, self._V_xi, self._V_r = mu_xi, V_xi, V_r
        self._tau0_sq, self._tau1_sq = tau0_sq, tau1_sq
        self._a_phi, self._b_phi = a_phi, b_phi
        self._Jf = tf.cast(tf.shape(X)[1], tf.float32)
        self._use_tmh = use_tmh

    @property
    def is_calibrated(self):
        return True

    def one_step(self, state, previous_kernel_results):
        beta, r, xi, eta, gamma, phi = state
        X, q, v_draws, rc_indices = self._X, self._q, self._v_draws, self._rc_indices
        h_offset = self._h_offset

        # 1. beta_bar (TMH or RWMH)
        def lp_beta(beta_):
            return (_ll_total(beta_, r, xi, eta, X, q, v_draws, rc_indices,
                              h_offset)
                    - 0.5 * tf.reduce_sum((beta_ - self._mu_beta) ** 2 / self._V_beta))
        if self._use_tmh:
            beta, acc_beta = tmh_step(beta, lp_beta, self._kappa_beta)
        else:
            beta, acc_beta = rwmh_step(beta, lp_beta, self._kappa_beta)

        # 2. r (RWMH)
        def lp_r(r_):
            return (_ll_total(beta, r_, xi, eta, X, q, v_draws, rc_indices,
                              h_offset)
                    - 0.5 * tf.reduce_sum(r_ ** 2 / self._V_r))
        r, acc_r = rwmh_step(r, lp_r, self._kappa_r)

        # 3. xi_bar (RWMH, batched over T)
        def lp_xi(xi_):
            return (_ll_per_market(beta, r, xi_, eta, X, q, v_draws, rc_indices,
                                   h_offset)
                    - 0.5 * (xi_ - self._mu_xi) ** 2 / self._V_xi)
        xi, acc_xi = rwmh_step(xi, lp_xi, self._kappa_xi)

        # 4. eta (TMH or RWMH per market)
        var_eta = gamma * self._tau1_sq + (1.0 - gamma) * self._tau0_sq
        h_offsets = h_offset if h_offset is not None else tf.zeros_like(eta)
        if self._use_tmh:
            def _mh_one_market(args):
                eta_t, xi_t, X_t, q_t, var_eta_t, h_t = args

                def lp_eta(eta_):
                    s = _shares_single(beta, r, xi_t, eta_, X_t, v_draws,
                                       rc_indices, h_t)
                    return (tf.reduce_sum(q_t * tf.math.log(s))
                            - 0.5 * tf.reduce_sum(eta_ ** 2 / var_eta_t))
                new_eta, acc = tmh_step(eta_t, lp_eta, self._kappa_eta)
                return new_eta, tf.cast(acc, tf.float32)
        else:
            def _mh_one_market(args):
                eta_t, xi_t, X_t, q_t, var_eta_t, h_t = args

                def lp_eta(eta_):
                    s = _shares_single(beta, r, xi_t, eta_, X_t, v_draws,
                                       rc_indices, h_t)
                    return (tf.reduce_sum(q_t * tf.math.log(s))
                            - 0.5 * tf.reduce_sum(eta_ ** 2 / var_eta_t))
                new_eta, acc = rwmh_step(eta_t, lp_eta, self._kappa_eta)
                return new_eta, tf.cast(acc, tf.float32)
        new_eta, acc_eta_vec = tf.map_fn(
            _mh_one_market, (eta, xi, X, q, var_eta, h_offsets),
            fn_output_signature=(tf.float32, tf.float32))
        eta = new_eta
        acc_eta = tf.reduce_mean(acc_eta_vec)

        # 5. gamma (closed-form Bernoulli)
        gamma = sample_gamma(eta, phi, self._tau0_sq, self._tau1_sq)

        # 6. phi (closed-form Beta)
        phi = sample_phi(gamma, self._a_phi, self._b_phi, self._Jf)

        return (beta, r, xi, eta, gamma, phi), Lu25KernelResults(
            tf.cast(acc_beta, tf.float32),
            tf.reduce_mean(tf.cast(acc_r, tf.float32)),
            tf.reduce_mean(tf.cast(acc_xi, tf.float32)), acc_eta)

    def bootstrap_results(self, init_state):
        return Lu25KernelResults(0.0, 0.0, 0.0, 0.0)


class Lu25Model(BayesianChoiceModel):
    """Bayesian shrinkage estimator (Lu & Shimizu 2025).

    Args:
        tau0_sq: spike variance
        tau1_sq: slab variance
        a_phi, b_phi: Beta prior for phi
        mu_beta, V_beta: prior mean/variance for beta_bar
        mu_xi, V_xi: prior mean/variance for xi_bar
        V_r: prior variance for r
        rc_indices: indices of covariates with random coefficients
        R0: number of MC draws
        n_mcmc: total MCMC iterations
        n_burnin: burn-in iterations
    """

    def __init__(self, tau0_sq=1e-3, tau1_sq=1.0, a_phi=1.0, b_phi=1.0,
                 mu_beta=0.0, V_beta=10.0, mu_xi=0.0, V_xi=10.0, V_r=0.5,
                 rc_indices=None, R0=200, n_mcmc=5000, n_burnin=1500,
                 use_tmh=True):
        super().__init__(n_mcmc=n_mcmc, n_burnin=n_burnin, R0=R0)
        self.tau0_sq, self.tau1_sq = tau0_sq, tau1_sq
        self.a_phi, self.b_phi = a_phi, b_phi
        self.mu_beta, self.V_beta = mu_beta, V_beta
        self.mu_xi, self.V_xi = mu_xi, V_xi
        self.V_r = V_r
        self.rc_indices = rc_indices
        self.use_tmh = use_tmh

    def _prepare_data(self, X, q):
        self.X = tf.constant(X, dtype=tf.float32)
        self.q = tf.constant(q, dtype=tf.float32)
        self.T, self.J, self.d_X = self.X.shape
        if self.rc_indices is None:
            self.rc_indices = list(range(self.d_X))
        self._rc = tf.constant(self.rc_indices, dtype=tf.int32)
        self.d_rc = len(self.rc_indices)

    def _simple_logit_ols(self):
        """Initialize beta_bar, xi_bar via simple logit OLS.

        Solves: log(s_j / s_0) = X' beta + xi_t (with market dummies).

        Returns:
            beta_init: (d_X,) initial mean taste parameters
            xi_init: (T,) initial market intercepts
        """
        shares = self.q / tf.reduce_sum(self.q, axis=1, keepdims=True)
        log_share_ratio = (
            tf.math.log(tf.clip_by_value(shares[:, 1:], 1e-10, 1.0))
            - tf.math.log(tf.clip_by_value(shares[:, 0:1], 1e-10, 1.0))
        )  # (T, J)

        features_flat = tf.reshape(self.X, (-1, self.d_X))        # (T*J, d_X)
        market_dummies = tf.repeat(tf.eye(self.T), self.J, axis=0)  # (T*J, T)
        design_matrix = tf.concat([features_flat, market_dummies], axis=1)  # (T*J, d_X+T)
        target = tf.reshape(log_share_ratio, (-1,))                # (T*J,)

        coefficients = tf.linalg.lstsq(
            design_matrix, target[:, tf.newaxis])[:, 0]
        beta_init = coefficients[:self.d_X]
        xi_init = coefficients[self.d_X:]
        return beta_init, xi_init

    def _init_state(self):
        self.v_draws = tf.random.normal((self.R0, self.d_rc))
        beta_init, xi_init = self._simple_logit_ols()
        self._init_state_tuple = (
            beta_init,                        # beta_bar
            tf.zeros(self.d_rc),              # r (log std of RC)
            xi_init,                          # xi_bar
            tf.zeros((self.T, self.J)),       # eta (all sparse)
            tf.zeros((self.T, self.J)),       # gamma (all sparse)
            tf.fill((self.T,), 0.1),          # phi (low inclusion prob)
        )
        self.kappa_beta = 2.38 / np.sqrt(self.d_X)
        self.kappa_eta = 2.38 / np.sqrt(self.J)
        self.kappa_r = 0.01
        self.kappa_xi = 0.01

    def _make_kernel(self):
        return Lu25MCMCKernel(
            self.X, self.q, self.v_draws, self._rc,
            self.kappa_beta, self.kappa_r, self.kappa_xi, self.kappa_eta,
            *[tf.constant(v, tf.float32) for v in [
                self.mu_beta, self.V_beta, self.mu_xi, self.V_xi, self.V_r,
                self.tau0_sq, self.tau1_sq, self.a_phi, self.b_phi]],
            use_tmh=self.use_tmh)

    def _get_kappa_blocks(self, kernel):
        return [("beta", kernel._kappa_beta, "kappa_beta"),
                ("r", kernel._kappa_r, "kappa_r"),
                ("xi", kernel._kappa_xi, "kappa_xi"),
                ("eta", kernel._kappa_eta, "kappa_eta")]

    def _extract_samples(self, chain, trace):
        beta, r, xi, eta, gamma, phi = chain
        self.samples_ = {
            "beta_bar": beta.numpy(), "r": r.numpy(), "xi_bar": xi.numpy(),
            "eta": eta.numpy(), "gamma": gamma.numpy(), "phi": phi.numpy()}

    def compute_shares(self, X):
        """Predict shares using posterior means.

        Args:
            X: (T, J, d_X) product characteristics
        """
        beta_bar = tf.constant(np.mean(self.samples_["beta_bar"], 0), tf.float32)
        r = tf.constant(np.mean(self.samples_["r"], 0), tf.float32)
        xi_bar = tf.constant(np.mean(self.samples_["xi_bar"], 0), tf.float32)
        eta = tf.constant(np.mean(self.samples_["eta"], 0), tf.float32)
        return _shares_all(beta_bar, r, xi_bar, eta,
                           tf.constant(X, tf.float32),
                           self.v_draws, self._rc).numpy()

    def get_posterior_summary(self):
        """Posterior summary including sigma = exp(r)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        def _summarize(x):
            return {"mean": np.mean(x, 0), "std": np.std(x, 0),
                    "ci_lower": np.percentile(x, 2.5, 0),
                    "ci_upper": np.percentile(x, 97.5, 0)}

        return {"beta_bar": _summarize(self.samples_["beta_bar"]),
                "r": _summarize(self.samples_["r"]),
                "sigma": _summarize(np.exp(self.samples_["r"])),
                "xi_bar": _summarize(self.samples_["xi_bar"])}
