"""Simulation DGPs for model validation.

Contains data generating processes for:
  - Lu & Shimizu (2025): DGP 1-4 (sparse/non-sparse x exogenous/endogenous)
  - DeepHalo + Shrinkage: context/no-context x sparse/non-sparse
"""

import numpy as np


def simulate_lu25_dgp(T, J, N=1000, dgp_type="dgp1", seed=42):
    """Generate synthetic data from Lu & Shimizu (2025) Section 4.1 DGPs.

    4 DGP configurations (2x2 design):
      DGP1: sparse eta, exogenous price
      DGP2: sparse eta, endogenous price
      DGP3: non-sparse eta, exogenous price
      DGP4: non-sparse eta, endogenous price

    True parameters: beta_p=-1, beta_w=0.5, sigma=1.5, xi_bar=-1.

    Args:
        T: number of markets
        J: number of products per market
        N: number of consumer draws per market
        dgp_type: one of "dgp1", "dgp2", "dgp3", "dgp4"
        seed: random seed

    Returns:
        X: product characteristics (T, J, 2) -- [price, w]
        q: choice counts (T, J+1) -- including outside option
        true_params: dict of true parameter values
    """
    rng = np.random.default_rng(seed)
    beta_p, beta_w, sigma_p, xi_bar_true = -1.0, 0.5, 1.5, -1.0
    sparse = dgp_type in ("dgp1", "dgp2")
    endogenous = dgp_type in ("dgp2", "dgp4")

    w = rng.uniform(1, 2, size=(T, J))
    eta = np.zeros((T, J))
    gamma = np.zeros((T, J))
    if sparse:
        for t in range(T):
            for j in range(int(np.floor(0.4 * J))):
                gamma[t, j] = 1.0
                eta[t, j] = 1.0 if (j + 1) % 2 == 1 else -1.0
    else:
        eta = rng.normal(0, 1.0 / 3.0, size=(T, J))
        gamma = None

    xi = xi_bar_true + eta
    alpha = np.zeros((T, J))
    if endogenous:
        if sparse:
            alpha[eta == 1.0], alpha[eta == -1.0] = 0.3, -0.3
        else:
            alpha[eta >= 1.0 / 3.0], alpha[eta <= -1.0 / 3.0] = 0.3, -0.3

    cost_shock = rng.normal(0, 0.7, size=(T, J))
    p = alpha + 0.3 * w + cost_shock
    X = np.stack([p, w], axis=-1).astype(np.float32)

    # Compute shares via MC integration of eq (11):
    # s_jt = (1/N) sum_i exp(d_jt + mu_ijt) / (1 + sum_k exp(d_kt + mu_ikt))
    # where mu_ijt = sigma * p_jt * v_i, v_i ~ N(0,1)
    delta = beta_p * p + beta_w * w + xi
    v_draws = rng.standard_normal(N)
    mu = sigma_p * p[:, :, np.newaxis] * v_draws[np.newaxis, np.newaxis, :]
    util = delta[:, :, np.newaxis] + mu
    util_all = np.concatenate([np.zeros((T, 1, N)), util], axis=1)
    max_u = np.max(util_all, axis=1, keepdims=True)
    exp_u = np.exp(util_all - max_u)
    shares = np.mean(exp_u / np.sum(exp_u, axis=1, keepdims=True), axis=2)
    q = (shares * N).astype(np.float32)

    return X, q, {
        "beta_bar": np.array([beta_p, beta_w], np.float32),
        "sigma": np.array([sigma_p], np.float32),
        "r": np.array([np.log(sigma_p)], np.float32),
        "xi_bar": xi_bar_true,
        "eta": eta.astype(np.float32),
        "gamma": gamma.astype(np.float32) if gamma is not None else None,
        "cost_shock": cost_shock.astype(np.float32),
    }


def _compute_context_effect(p, alpha, context_type):
    """Compute context effect based on type.

    Args:
        p: (T, J) prices
        alpha: context effect strength
        context_type: one of "quadratic", "rank", "asymmetric"

    Returns:
        context_effect: (T, J) context effect values
    """
    p_bar = np.mean(p, axis=1, keepdims=True)

    if context_type == "quadratic":
        # alpha * (p_j - p_bar)^2 -- symmetric penalty for price deviation
        return alpha * (p - p_bar) ** 2

    elif context_type == "rank":
        # alpha * rank(p_j) / J -- price rank effect (discontinuous)
        # rank 1 = cheapest, rank J = most expensive
        T, J = p.shape
        ranks = np.zeros_like(p)
        for t in range(T):
            ranks[t] = np.argsort(np.argsort(p[t])) + 1  # 1-indexed ranks
        return alpha * ranks / J

    elif context_type == "asymmetric":
        # alpha * max(p_j - p_bar, 0) -- only penalizes above-average prices
        return alpha * np.maximum(p - p_bar, 0)

    else:
        raise ValueError(f"Unknown context_type: {context_type}")


def simulate_deephalo_shrinkage_dgp(T, J, N=1000, dgp_type="context_sparse",
                                    alpha=None, context_type="quadratic",
                                    seed=42):
    """Generate data with context-dependent utility + sparse demand shocks.

    True utility (standard logit, no random coefficients):
        delta_jt = beta_p*p + beta_w*w + context(p, S_t) + xi_bar + eta_jt

    Context effect types (all nonlinear, cannot be decomposed into beta*p + xi_bar):
        quadratic:  alpha * (p_j - p_bar)^2      -- symmetric deviation penalty (smooth)
        rank:       alpha * rank(p_j) / J        -- price rank effect (discontinuous)
        asymmetric: alpha * max(p_j - p_bar, 0)  -- above-average penalty only

    4 DGP configurations (2x2):
        context_sparse:      context effect + sparse eta
        context_nonsparse:   context effect + non-sparse eta
        nocontext_sparse:    no context + sparse eta
        nocontext_nonsparse: no context + non-sparse eta

    Args:
        T: number of markets
        J: number of products per market
        N: number of consumers per market
        dgp_type: DGP configuration name
        alpha: context effect strength (overrides dgp_type default if given)
        context_type: type of context effect ("quadratic", "rank", "asymmetric")
        seed: random seed

    Returns:
        X: product features (T, J, 2) -- [price, w]
        q: choice counts (T, J+1) -- including outside option
        true_params: dict of true parameter values
    """
    rng = np.random.default_rng(seed)
    beta_p, beta_w, xi_bar_true = -1.0, 0.5, -1.0
    if alpha is None:
        alpha = 0.3 if dgp_type.startswith("context") else 0.0
    sparse = "sparse" in dgp_type and "nonsparse" not in dgp_type

    w = rng.uniform(1, 2, size=(T, J))
    cost_shock = rng.normal(0, 0.7, size=(T, J))
    p = 0.3 * w + cost_shock

    eta = np.zeros((T, J))
    gamma_true = np.zeros((T, J))
    if sparse:
        for t in range(T):
            for j in range(int(np.floor(0.4 * J))):
                gamma_true[t, j] = 1.0
                eta[t, j] = 1.0 if (j + 1) % 2 == 1 else -1.0
    else:
        gamma_true[:] = 1
        eta = rng.normal(0, 1.0 / 3.0, size=(T, J))

    context_effect = _compute_context_effect(p, alpha, context_type)
    delta = beta_p * p + beta_w * w + context_effect + xi_bar_true + eta

    util_all = np.concatenate([np.zeros((T, 1)), delta], axis=1)
    max_u = np.max(util_all, axis=1, keepdims=True)
    exp_u = np.exp(util_all - max_u)
    shares = exp_u / np.sum(exp_u, axis=1, keepdims=True)
    q = (shares * N).astype(np.float32)

    X = np.stack([p, w], axis=-1).astype(np.float32)
    return X, q, {
        "beta_p": beta_p, "beta_w": beta_w,
        "alpha": alpha, "context_type": context_type,
        "xi_bar": xi_bar_true,
        "eta": eta.astype(np.float32),
        "gamma": gamma_true.astype(np.float32),
        "context_effect": context_effect.astype(np.float32),
        "shares": shares.astype(np.float32),
    }


def simulate_parameter_recovery_dgp(T=50, J=10, N=1000, phi_true=0.4, seed=42):
    """DGP for DeepHalo+xi parameter recovery experiment.

    Standard logit (no RC, no context effects):
        v_jt = beta_price * price_jt + beta_quality * quality_jt + xi_bar_t + eta_jt

    True parameters:
        beta = [-1.0, 0.3]
        xi_bar_t ~ N(-1, 0.3) per market
        eta: phi_true fraction active with eta in {+1, -1}, rest zero
        phi = phi_true

    Args:
        T: number of markets
        J: number of products per market
        N: consumers per market (for multinomial draw)
        phi_true: true sparsity inclusion probability
        seed: random seed

    Returns:
        X: (T, J, 2) product features [price, quality]
        q: (T, J+1) choice counts including outside option
        true_params: dict of true parameter values
    """
    rng = np.random.default_rng(seed)
    beta_price, beta_quality = -1.0, 0.3

    price = rng.uniform(1, 5, size=(T, J))
    quality = rng.standard_normal(size=(T, J))
    xi_bar = rng.normal(-1.0, 0.3, size=(T,))

    n_active = int(np.floor(phi_true * J))
    eta = np.zeros((T, J))
    gamma_true = np.zeros((T, J))
    for t in range(T):
        gamma_true[t, :n_active] = 1.0
        for j in range(n_active):
            eta[t, j] = 1.0 if (j + 1) % 2 == 1 else -1.0

    v = beta_price * price + beta_quality * quality + xi_bar[:, np.newaxis] + eta
    v_all = np.concatenate([np.zeros((T, 1)), v], axis=1)
    max_v = np.max(v_all, axis=1, keepdims=True)
    exp_v = np.exp(v_all - max_v)
    shares = exp_v / np.sum(exp_v, axis=1, keepdims=True)

    q = np.zeros((T, J + 1), dtype=np.float32)
    for t in range(T):
        q[t] = rng.multinomial(N, shares[t]).astype(np.float32)

    X = np.stack([price, quality], axis=-1).astype(np.float32)
    return X, q, {
        "beta": np.array([beta_price, beta_quality], dtype=np.float32),
        "xi_bar": xi_bar.astype(np.float32),
        "eta": eta.astype(np.float32),
        "gamma": gamma_true.astype(np.float32),
        "phi": phi_true,
    }
