"""BLP estimator using PyBLP library.

Two IV configs (Lu & Shimizu 2025, Section 4):
  - "cost_iv": excluded Z = (w^2, u, u^2), exogenous (1, w) auto-added
  - "weak_iv": excluded Z = (w^2, w^3, w^4), exogenous (1, w) auto-added
"""

import numpy as np
import pandas as pd
import pyblp


class PyBLPEstimator:
    """BLP estimator via PyBLP.

    Args:
        iv_type: "cost_iv" or "weak_iv"
        n_draws: Monte Carlo draws for integration (default 1000)
        sigma_init: initial value for sigma optimization
        seed: random seed for MC draws
    """

    def __init__(self, iv_type="cost_iv", n_draws=1000,
                 sigma_init=1.0, seed=42):
        self.iv_type = iv_type
        self.n_draws = n_draws
        self.sigma_init = sigma_init
        self.seed = seed
        self.is_fitted = False

    def _build_product_data(self, X, q, cost_shock):
        """Convert (T,J,2) arrays to pyblp product_data DataFrame.

        Args:
            X: (T, J, 2) product characteristics [price, w]
            q: (T, J+1) choice counts including outside option
            cost_shock: (T, J) cost shocks, required for iv_type="cost_iv"
        """
        T, J = X.shape[0], X.shape[1]
        shares = q / q.sum(axis=1, keepdims=True)
        s_inside = shares[:, 1:]
        prices = X[:, :, 0]
        w = X[:, :, 1]

        data = {
            'market_ids': np.repeat(np.arange(T), J),
            'shares': s_inside.ravel(),
            'prices': prices.ravel(),
            'w': w.ravel(),
        }

        # Excluded instruments only; PyBLP auto-adds exogenous X1 vars (1, w)
        w_flat = w.ravel()
        if self.iv_type == "cost_iv":
            u_flat = cost_shock.ravel()
            data['demand_instruments0'] = w_flat ** 2
            data['demand_instruments1'] = u_flat
            data['demand_instruments2'] = u_flat ** 2
        elif self.iv_type == "weak_iv":
            data['demand_instruments0'] = w_flat ** 2
            data['demand_instruments1'] = w_flat ** 3
            data['demand_instruments2'] = w_flat ** 4
        else:
            raise ValueError(f"Unknown iv_type: {self.iv_type}")

        return pd.DataFrame(data)

    def fit(self, X, q, cost_shock=None):
        """Estimate BLP model.

        Args:
            X: (T, J, 2) product characteristics [price, w]
            q: (T, J+1) choice counts including outside option
            cost_shock: (T, J) required if iv_type="cost_iv"
        """
        if self.iv_type == "cost_iv" and cost_shock is None:
            raise ValueError("cost_shock required for iv_type='cost_iv'")

        self.T, self.J = X.shape[0], X.shape[1]
        df = self._build_product_data(X, q, cost_shock)

        problem = pyblp.Problem(
            product_formulations=(
                pyblp.Formulation('1 + prices + w'),
                pyblp.Formulation('0 + prices'),
            ),
            product_data=df,
            integration=pyblp.Integration('monte_carlo', self.n_draws,
                                          {'seed': self.seed}),
        )

        results = problem.solve(
            sigma=np.array([[self.sigma_init]]),
        )

        self._results = results
        self.xi_bar_hat = float(results.beta[0])
        self.beta_hat = results.beta[1:].flatten()
        self.sigma_hat = float(results.sigma[0, 0])
        eta_hat = results.xi.reshape(self.T, self.J)
        self.xi_hat = self.xi_bar_hat + eta_hat
        self.is_fitted = True
        return self

    def get_results(self, true_params=None):
        """Return estimates dict, optionally with biases vs true_params.

        Args:
            true_params: dict from simulate_lu25_dgp (optional)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        results = {
            "xi_bar_est": float(self.xi_bar_hat),
            "beta_p_est": float(self.beta_hat[0]),
            "beta_w_est": float(self.beta_hat[1]),
            "sigma_est": float(self.sigma_hat),
        }

        if true_params is not None:
            beta_true = true_params["beta_bar"]
            sigma_true = true_params["sigma"][0]
            xi_bar_true = true_params["xi_bar"]
            xi_true = xi_bar_true + true_params["eta"]

            results.update({
                "xi_bar_bias": float(self.xi_bar_hat - xi_bar_true),
                "beta_p_bias": float(self.beta_hat[0] - beta_true[0]),
                "beta_w_bias": float(self.beta_hat[1] - beta_true[1]),
                "sigma_bias": float(self.sigma_hat - sigma_true),
                "xi_jt_abs_bias": float(np.mean(np.abs(self.xi_hat - xi_true))),
                "xi_jt_sd": float(np.std(self.xi_hat)),
            })

        return results
