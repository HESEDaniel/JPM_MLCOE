"""DGP signal diagnostics: macro correlation, decoy, and similarity effect magnitudes."""

from __future__ import annotations

import numpy as np


def sanity_check(data: dict) -> dict:
    """Numeric diagnostics for the macro, decoy, and similarity signals.

    Recomputes the three structural signals the report claims the DGP
    encodes, directly from the simulated ground truth: (1) the macro signal,
    where the chosen offer's mean psi correlates with the latent state x_t
    and is itself persistent over time; (2) the decoy signal, comparing the
    chosen offer's discount against the same-category mean discount on its
    slate; and (3) the similarity signal, the rate at which the chosen offer
    falls in a same-(category, discount) twin group on its slate.

    Parameters
    ----------
    data : dict
        Output of ``simulate_macro_choice_dgp``. Uses the public keys
        ``slate_indicator``, ``choice``, ``T``, ``N_t`` and the ground-truth
        keys ``_x_true`` (latent state), ``_psi_offer`` (per-offer psi),
        ``_cat`` (offer category), and ``_disc`` (offer discount).

    Returns
    -------
    dict
        Named scalar diagnostics: macro correlation ``corr(mean chosen
        psi_j, x_t)`` and its lag-1 autocorrelation, the decoy signal (chosen
        discount minus slate same-category mean discount), the similarity
        collision rate, and summary statistics for x_t and mean slate size.
    """
    slate_indicator = data["slate_indicator"].astype(np.float32)
    choice = data["choice"]
    x = data["_x_true"]
    psi_offer = data["_psi_offer"]
    cat = data["_cat"]
    disc = data["_disc"]

    slate_size = slate_indicator.sum(axis=-1)

    # 1. Macro signal: chosen psi_j correlates with x_t
    chosen_psi = psi_offer[choice]
    mean_chosen_psi = chosen_psi.mean(axis=-1)
    corr_psi_x = float(np.corrcoef(mean_chosen_psi, x)[0, 1])
    ac1 = float(np.corrcoef(mean_chosen_psi[:-1], mean_chosen_psi[1:])[0, 1])

    # 2. Decoy signal: chosen offer's disc vs slate same-cat mean disc.
    T, N_t = data["T"], data["N_t"]
    chosen_disc = disc[choice]
    chosen_cat = cat[choice]
    decoy_signal_vals = []
    for t in range(T):
        for i in range(N_t):
            S_idx = np.where(slate_indicator[t, i] > 0)[0]
            same_cat_in_S = S_idx[cat[S_idx] == chosen_cat[t, i]]
            if len(same_cat_in_S) > 1:
                decoy_signal_vals.append(chosen_disc[t, i] - disc[same_cat_in_S].mean())
    decoy_signal = float(np.mean(decoy_signal_vals)) if decoy_signal_vals else float("nan")

    # 3. Similarity signal: how often the chosen offer sits in a same-(cat, disc) twin group.
    sim_collision_rates = []
    for t in range(T):
        for i in range(N_t):
            S_idx = np.where(slate_indicator[t, i] > 0)[0]
            chosen_idx = choice[t, i]
            n_same_cd = ((cat[S_idx] == cat[chosen_idx]) & (disc[S_idx] == disc[chosen_idx])).sum()
            sim_collision_rates.append(int(n_same_cd >= 2))   # >=2 includes chosen itself
    sim_collision_rate = float(np.mean(sim_collision_rates))

    return {
        "corr(mean chosen psi_j, x_t)": corr_psi_x,
        "AC(1) of mean chosen psi_j": ac1,
        "decoy signal (chosen disc - slate same-cat mean disc)": decoy_signal,
        "similarity collision rate (chosen in same-cat-disc group)": sim_collision_rate,
        "x_t empirical mean": float(x.mean()),
        "x_t empirical std": float(x.std()),
        "mean slate size K": float(slate_size.mean()),
    }
