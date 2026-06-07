"""Synthetic DGP - universe + per-offer ground truth + AR(1) macro + halo.

Per offer j (frozen at experiment init):
    base_j ~ N(0, base_scale**2)    intrinsic appeal
    psi_j  ~ N(0, psi_scale**2)     macro loading
    cat_j  in {0, ..., C-1}         latent category (hidden from Featureless models)
    disc_j in {0, ..., D-1}         discount level (hidden)

State trajectory (AR(1), stationary init):
    x_t = mu + phi * x_{t-1} + eps_t,    eps_t ~ N(0, sigma^2)

Per (t, i) -- repeated cross-section, independent across customers:
    Slate S ~ Uniform_{K-subsets}({0,...,M-1}),  |S| ~ Uniform{K_min..K_max}.

    Halo (Zhang 2025 decoy + similarity effects, restricted to same-category
    pairs of the slate). For each j in S let
        n_lower_j  = #{k in S\\{j} : cat_k = cat_j, disc_k < disc_j}
        n_higher_j = #{k in S\\{j} : cat_k = cat_j, disc_k > disc_j}
        n_equal_j  = #{k in S\\{j} : cat_k = cat_j, disc_k = disc_j}

        halo_j = beta_decoy * (n_lower_j - n_higher_j)        # decoy effect
                 - beta_similar * n_equal_j                    # similarity effect
                 - beta_quad * n_equal_j**2                    # quadratic similarity
                                                              # (optional nonlinear knob)

    Utility and choice (Gumbel-max RUM, equivalent to softmax categorical):
        V_j = base_j + psi_j * x_t + halo_j
        c_{ti} = argmax_{j in S} (V_j + Gumbel(0, 1))

Scenarios (psi_scale, beta_decoy, beta_similar):
    A : (0.5,  0.0, 0.0)  -- macro only, no halo
    B : (0.5,  0.5, 0.5)  -- main scenario
    C : (0.15, 0.5, 0.5)  -- weak macro
    D : (0.5,  1.0, 1.0)  -- strong halo
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import numpy as np


SCENARIO_PARAMS = {
    "A": dict(psi_scale=0.5,  beta_decoy=0.0, beta_similar=0.0),
    "B": dict(psi_scale=0.5,  beta_decoy=0.5, beta_similar=0.5),
    "C": dict(psi_scale=0.15, beta_decoy=0.5, beta_similar=0.5),
    "D": dict(psi_scale=0.5,  beta_decoy=1.0, beta_similar=1.0),
}


@dataclass
class DGPConfig:
    # Universe + slate
    M: int = 20          # universe size
    C: int = 3           # number of latent categories
    D: int = 3           # number of discount levels
    K_min: int = 2
    K_max: int = 5
    # State equation
    mu: float = 0.0
    phi: float = 0.9
    sigma: float = 0.5
    # Per-offer ground truth
    psi_scale: float = 0.5
    base_scale: float = 1.0
    # Halo coefficients (decoy + similarity)
    beta_decoy: float = 0.5      # decoy: same-cat lower-disc competitors boost an offer
    beta_similar: float = 0.5    # penalty for same-cat same-disc competitors
    beta_quad: float = 0.0       # optional quadratic similarity: -beta_quad * n_equal**2
    # Data scale
    T: int = 300
    N_t: int = 50
    # Outside (no-choice) option: choice index M signals "no activation".
    outside_option: bool = False
    # Macro loading of the outside option: u_outside = outside_macro * x_t.
    outside_macro: float = 0.0


def simulate_macro_choice_dgp(
    scenario: Optional[Literal["A", "B", "C", "D"]] = "B",
    cfg: Optional[DGPConfig] = None,
    seed: int = 42,
    x_true_override: Optional[np.ndarray] = None,
) -> dict:
    """Generate one synthetic macro-choice dataset.

    Parameters
    ----------
    scenario : {"A", "B", "C", "D"} or None
        Named scenario whose (psi_scale, beta_decoy, beta_similar) overrides
        the corresponding fields of ``cfg``. If None, ``cfg`` is used as-is.
    cfg : DGPConfig, optional
        Generative-process configuration. Defaults to ``DGPConfig()``.
    seed : int
        Seed for the NumPy random generator.
    x_true_override : np.ndarray, optional
        Length-T external macro state to use instead of simulating an AR(1)
        trajectory (e.g. to inject a non-AR(1) ``x_t``).

    Returns
    -------
    dict
        Observable keys:
          'slate_indicator' (T, N_t, M)  uint8
          'choice'          (T, N_t)     int64  - universe index
          'M, C, D, K_min, K_max, T, N_t, scenario'
        Hidden (verification only) keys:
          '_x_true'  (T,),  '_base'/'_psi_offer' (M,),
          '_cat'/'_disc' (M,) int8,  '_true_params' dict
    """
    cfg = cfg or DGPConfig()
    if scenario is not None:
        for k, v in SCENARIO_PARAMS[scenario].items():
            setattr(cfg, k, v)

    rng = np.random.default_rng(seed)
    M, C, D = cfg.M, cfg.C, cfg.D
    K_min, K_max = cfg.K_min, cfg.K_max
    T, N_t = cfg.T, cfg.N_t
    mu, phi, sigma = cfg.mu, cfg.phi, cfg.sigma
    beta_decoy, beta_similar, beta_quad = cfg.beta_decoy, cfg.beta_similar, cfg.beta_quad
    outside_option = cfg.outside_option
    outside_macro = cfg.outside_macro

    assert 1 <= K_min <= K_max <= M, (
        f"slate size range must satisfy 1 <= K_min ({K_min}) <= K_max ({K_max}) <= M ({M})"
    )

    # 0. Frozen per-offer ground truth
    base = rng.normal(0.0, cfg.base_scale, size=M).astype(np.float32)
    psi_offer = rng.normal(0.0, cfg.psi_scale, size=M).astype(np.float32)
    cat = rng.integers(0, C, size=M).astype(np.int8)
    disc = rng.integers(0, D, size=M).astype(np.int8)

    # 1. State trajectory: AR(1), or an external override (e.g. a non-AR(1) x_t).
    if x_true_override is not None:
        x_true = np.asarray(x_true_override, dtype=np.float32)
        assert x_true.shape == (T,), f"x_true_override must be (T,)={T}, got {x_true.shape}"
    else:
        x_true = np.empty(T, dtype=np.float32)
        x_true[0] = rng.normal(mu / (1.0 - phi), sigma / np.sqrt(1.0 - phi**2))
        for t in range(1, T):
            x_true[t] = mu + phi * x_true[t - 1] + rng.normal(0.0, sigma)

    # 2. Per (t, i): variable-K slate + choice
    slate_indicator = np.zeros((T, N_t, M), dtype=np.uint8)
    choice = np.empty((T, N_t), dtype=np.int64)
    true_probs = np.zeros((T, N_t, M), dtype=np.float32)   # ground-truth softmax probs per slate

    universe = np.arange(M)
    for t in range(T):
        x_t = x_true[t]
        for i in range(N_t):
            K_ti = int(rng.integers(K_min, K_max + 1))
            S = rng.choice(universe, size=K_ti, replace=False)
            S_cat = cat[S]
            S_disc = disc[S]

            same_cat = (S_cat[:, None] == S_cat[None, :])
            np.fill_diagonal(same_cat, False)
            lower  = same_cat & (S_disc[None, :] < S_disc[:, None])
            higher = same_cat & (S_disc[None, :] > S_disc[:, None])
            equal  = same_cat & (S_disc[None, :] == S_disc[:, None])

            n_lower = lower.sum(axis=1).astype(np.float32)
            n_higher = higher.sum(axis=1).astype(np.float32)
            n_equal = equal.sum(axis=1).astype(np.float32)
            halo = (beta_decoy * (n_lower - n_higher)
                    - beta_similar * n_equal
                    - beta_quad * (n_equal ** 2)).astype(np.float32)
            V = base[S] + psi_offer[S] * x_t + halo

            slate_indicator[t, i, S] = 1
            if outside_option:
                # Append the outside alternative; index M = no activation.
                V_full = np.concatenate([V, [outside_macro * x_t]]).astype(np.float32)
                exp_V = np.exp(V_full - V_full.max())
                probs = exp_V / exp_V.sum()
                true_probs[t, i, S] = probs[:K_ti]              # in-slate mass sums to < 1
                gumbel = rng.gumbel(0.0, 1.0, size=K_ti + 1).astype(np.float32)
                pos = int(np.argmax(V_full + gumbel))
                choice[t, i] = M if pos == K_ti else S[pos]
            else:
                exp_V = np.exp(V - V.max())
                true_probs[t, i, S] = exp_V / exp_V.sum()
                gumbel = rng.gumbel(0.0, 1.0, size=K_ti).astype(np.float32)
                pos = int(np.argmax(V + gumbel))
                choice[t, i] = S[pos]

    return {
        "slate_indicator": slate_indicator,
        "choice": choice,
        "M": M, "C": C, "D": D,
        "K_min": K_min, "K_max": K_max,
        "T": T, "N_t": N_t,
        "scenario": scenario,
        "_x_true": x_true,
        "_base": base,
        "_psi_offer": psi_offer,
        "_cat": cat,
        "_disc": disc,
        "_true_probs": true_probs,     # (T, N_t, M) softmax(V_true) per slate
        "_true_params": asdict(cfg),
    }


def data_to_choice_dataset(
    data: dict, indices: np.ndarray | None = None,
    x_t_override: np.ndarray | None = None,
):
    """Convert a DGP dict to a choice-learn ChoiceDataset.

    Parameters
    ----------
    data : dict
        Output of ``simulate_macro_choice_dgp``.
    indices : np.ndarray, optional
        Flat (T*N_t,) indices selecting a train/val subset; if None, use all.
    x_t_override : np.ndarray, optional
        Length-T macro state used as ``shared_features`` in place of the DGP's
        true ``_x_true`` (e.g. to feed an imperfect proxy).

    Returns
    -------
    choice_learn.data.ChoiceDataset
        Dataset with shared macro features, slate availability, and choices.
    """
    from choice_learn.data import ChoiceDataset

    T, N_t, M = data["T"], data["N_t"], data["M"]
    B = T * N_t
    x_source = (np.asarray(x_t_override, dtype=np.float32)
                if x_t_override is not None else data["_x_true"])
    assert x_source.shape == (T,), f"x_t_override must be (T,)={T}, got {x_source.shape}"
    x_per_dec = np.repeat(x_source, N_t)[:, None].astype(np.float32)
    slate = data["slate_indicator"].reshape(B, M).astype(np.float32)
    choices = data["choice"].reshape(B).astype(np.int32)

    if indices is not None:
        x_per_dec, slate, choices = x_per_dec[indices], slate[indices], choices[indices]

    return ChoiceDataset(
        shared_features_by_choice=x_per_dec,
        available_items_by_choice=slate,
        choices=choices,
    )


