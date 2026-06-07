"""Unit tests for the synthetic data-generating process (decoy and similarity halo)."""

import numpy as np
import pytest

from src.datasets.dgp_macro import (
    DGPConfig,
    SCENARIO_PARAMS,
    simulate_macro_choice_dgp,
)
from src.datasets.dgp_sanity import sanity_check


@pytest.fixture(scope="module")
def small_dgp():
    return simulate_macro_choice_dgp(
        scenario="B", cfg=DGPConfig(M=10, C=3, D=3, K_min=2, K_max=4, T=30, N_t=10), seed=0
    )


# ---------- structural ----------


def test_output_keys(small_dgp):
    expected = {
        "slate_indicator", "choice",
        "M", "C", "D", "K_min", "K_max", "T", "N_t", "scenario",
        "_x_true", "_base", "_psi_offer", "_cat", "_disc", "_true_params",
    }
    assert expected.issubset(small_dgp.keys())


def test_shapes(small_dgp):
    M, C, D, T, N_t = small_dgp["M"], small_dgp["C"], small_dgp["D"], small_dgp["T"], small_dgp["N_t"]
    assert small_dgp["slate_indicator"].shape == (T, N_t, M)
    assert small_dgp["choice"].shape == (T, N_t)
    assert small_dgp["_x_true"].shape == (T,)
    assert small_dgp["_cat"].shape == (M,)
    assert small_dgp["_disc"].shape == (M,)


def test_slate_indicator_is_binary(small_dgp):
    assert set(np.unique(small_dgp["slate_indicator"]).tolist()) <= {0, 1}


def test_slate_size_in_range(small_dgp):
    sizes = small_dgp["slate_indicator"].sum(axis=-1)
    assert sizes.min() >= small_dgp["K_min"]
    assert sizes.max() <= small_dgp["K_max"]


def test_choice_is_in_slate(small_dgp):
    ind, c = small_dgp["slate_indicator"], small_dgp["choice"]
    for t in range(small_dgp["T"]):
        for i in range(small_dgp["N_t"]):
            assert ind[t, i, c[t, i]] == 1


def test_choice_valid_universe(small_dgp):
    c = small_dgp["choice"]
    assert c.min() >= 0 and c.max() < small_dgp["M"]


def test_categories_and_discs_in_range(small_dgp):
    assert 0 <= small_dgp["_cat"].min() and small_dgp["_cat"].max() < small_dgp["C"]
    assert 0 <= small_dgp["_disc"].min() and small_dgp["_disc"].max() < small_dgp["D"]


# ---------- variable K ----------


def test_slate_size_variation():
    cfg = DGPConfig(M=10, K_min=2, K_max=5, T=100, N_t=50)
    data = simulate_macro_choice_dgp(scenario="B", cfg=cfg, seed=0)
    sizes = data["slate_indicator"].sum(axis=-1)
    assert len(np.unique(sizes)) >= 3


def test_slate_size_fixed():
    cfg = DGPConfig(M=10, K_min=3, K_max=3, T=20, N_t=10)
    data = simulate_macro_choice_dgp(scenario="B", cfg=cfg, seed=0)
    sizes = data["slate_indicator"].sum(axis=-1)
    assert sizes.min() == 3 and sizes.max() == 3


# ---------- state trajectory ----------


def test_state_finite(small_dgp):
    x = small_dgp["_x_true"]
    assert np.isfinite(x).all() and np.abs(x).max() < 10


def test_state_persistence_long_run():
    data = simulate_macro_choice_dgp(scenario="B", cfg=DGPConfig(M=10, T=2000, N_t=2), seed=1)
    x = data["_x_true"]
    ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert ac1 > 0.75


# ---------- scenarios ----------


def test_scenario_params():
    for sc, p in SCENARIO_PARAMS.items():
        data = simulate_macro_choice_dgp(scenario=sc, cfg=DGPConfig(M=10, T=10, N_t=5), seed=0)
        tp = data["_true_params"]
        for k, v in p.items():
            assert tp[k] == v


def test_scenario_A_has_no_halo():
    data = simulate_macro_choice_dgp(scenario="A", cfg=DGPConfig(M=10, T=10, N_t=5), seed=0)
    assert data["_true_params"]["beta_decoy"] == 0.0
    assert data["_true_params"]["beta_similar"] == 0.0


# ---------- signal detection ----------


def test_macro_signal_DGP_B():
    data = simulate_macro_choice_dgp(scenario="B", cfg=DGPConfig(), seed=42)
    assert sanity_check(data)["corr(mean chosen psi_j, x_t)"] > 0.5


def test_macro_signal_weaker_in_DGP_C():
    cfg = DGPConfig()
    b = sanity_check(simulate_macro_choice_dgp(scenario="B", cfg=cfg, seed=42))
    c = sanity_check(simulate_macro_choice_dgp(scenario="C", cfg=cfg, seed=42))
    assert c["corr(mean chosen psi_j, x_t)"] < b["corr(mean chosen psi_j, x_t)"]


def test_decoy_signal_positive_when_beta_decoy_positive():
    cfg = DGPConfig()
    a = sanity_check(simulate_macro_choice_dgp(scenario="A", cfg=cfg, seed=42))
    b = sanity_check(simulate_macro_choice_dgp(scenario="B", cfg=cfg, seed=42))
    a_decoy = a["decoy signal (chosen disc - slate same-cat mean disc)"]
    b_decoy = b["decoy signal (chosen disc - slate same-cat mean disc)"]
    assert abs(a_decoy) < 0.1, f"DGP-A decoy {a_decoy:.3f} should be approx 0"
    assert b_decoy > 0.1, f"DGP-B decoy {b_decoy:.3f} should be > 0"


def test_similarity_collision_rate_drops_with_beta_similar():
    cfg = DGPConfig()
    a = sanity_check(simulate_macro_choice_dgp(scenario="A", cfg=cfg, seed=42))
    d = sanity_check(simulate_macro_choice_dgp(scenario="D", cfg=cfg, seed=42))
    a_rate = a["similarity collision rate (chosen in same-cat-disc group)"]
    d_rate = d["similarity collision rate (chosen in same-cat-disc group)"]
    assert d_rate < a_rate


def test_macro_preserved_when_halo_added():
    """Adding the halo should drop macro corr by less than 0.2."""
    cfg = DGPConfig()
    a = sanity_check(simulate_macro_choice_dgp(scenario="A", cfg=cfg, seed=42))
    b = sanity_check(simulate_macro_choice_dgp(scenario="B", cfg=cfg, seed=42))
    drop = a["corr(mean chosen psi_j, x_t)"] - b["corr(mean chosen psi_j, x_t)"]
    assert drop < 0.2


def test_sanity_check_returns_floats(small_dgp):
    assert all(isinstance(v, float) for v in sanity_check(small_dgp).values())
