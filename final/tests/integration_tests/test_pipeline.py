"""Integration test for the full pipeline: DGP -> train -> DPF + Adam -> PF (learned and oracle params) -> metrics.

Lightweight (~30s): T=50, N_t=10, 5 epochs, 10 Adam steps, 50 particles.
Verifies: whole pipeline works, no NaN, metrics finite, output shapes correct.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.datasets.dgp_macro import DGPConfig, data_to_choice_dataset, simulate_macro_choice_dgp
from src.inference import run_dpf_q2_adam, run_pf_q2
from src.choice_models.macro_deephalo import MacroDeepHalo


@pytest.fixture(scope="module")
def small_data():
    cfg = DGPConfig(M=10, K_min=2, K_max=5, T=50, N_t=10,
                    beta_decoy=0.5, beta_similar=0.5)
    return simulate_macro_choice_dgp(scenario="B", cfg=cfg, seed=42)


@pytest.fixture(scope="module")
def trained_model(small_data):
    """Train the DeepHalo on the small dataset."""
    tf.random.set_seed(42)
    ds = data_to_choice_dataset(small_data)
    model = MacroDeepHalo(M=small_data["M"],
                          epochs=5, batch_size=128, lr=1e-3, optimizer="adam")
    model.fit(ds, verbose=0)
    return model


def test_dgp_basic(small_data):
    """DGP outputs OK."""
    assert small_data["slate_indicator"].shape == (50, 10, 10)
    assert small_data["choice"].shape == (50, 10)
    # check each choice is within its slate
    si = small_data["slate_indicator"]
    ch = small_data["choice"]
    for t in range(50):
        for i in range(10):
            assert si[t, i, ch[t, i]] == 1, f"choice {ch[t,i]} not in slate at ({t},{i})"


def test_oracle_pf(small_data, trained_model):
    """Oracle-parameter PF state recovery."""
    tp = small_data["_true_params"]
    res = run_pf_q2(trained_model, small_data,
                     mu=tp["mu"], phi=tp["phi"], sigma=tp["sigma"],
                     n_particles=50, seed=42)
    T = small_data["T"]
    assert res["x_hat"].shape == (T,)
    assert res["x_std"].shape == (T,)
    assert res["ess"].shape == (T,)
    # No NaN
    assert np.isfinite(res["x_hat"]).all()
    assert np.isfinite(res["x_std"]).all()
    assert (res["x_std"] > 0).all()
    # RMSE finite + reasonable
    rmse = float(np.sqrt(np.mean((res["x_hat"] - small_data["_x_true"]) ** 2)))
    assert 0 < rmse < 5.0, f"RMSE {rmse} unreasonable"


def test_dpf_adam(small_data, trained_model):
    """DPF + Adam joint MLE."""
    res = run_dpf_q2_adam(trained_model, small_data,
                       free_psi=True, free_dh=True,
                       n_particles=50, n_steps=10, lr=0.05, seed=42, verbose=False)
    # Params OK
    assert np.isfinite(res["mu"])
    assert np.isfinite(res["phi"]) and -1 < res["phi"] < 1
    assert np.isfinite(res["sigma"]) and res["sigma"] > 0
    # psi_offer OK
    assert res["psi_offer"].shape == (small_data["M"],)
    assert np.isfinite(res["psi_offer"]).all()
    # History monotonic-ish (gradient descent)
    nll = res["history"]["neg_log_lik"]
    assert len(nll) == 10
    assert np.isfinite(nll).all()


def test_full_pipeline(small_data, trained_model):
    """Full pipeline: DPF + Adam, then PF with the learned params, then state recovery."""
    # DPF + Adam
    res_2c = run_dpf_q2_adam(trained_model, small_data,
                           free_psi=True, free_dh=True,
                           n_particles=50, n_steps=10, lr=0.05, seed=42, verbose=False)
    # PF with learned params
    res_A = run_pf_q2(trained_model, small_data,
                       mu=res_2c["mu"], phi=res_2c["phi"], sigma=res_2c["sigma"],
                       n_particles=50, seed=42)
    # All metrics finite
    err = res_A["x_hat"] - small_data["_x_true"]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    cov95 = float(np.mean(np.abs(err) <= 1.96 * res_A["x_std"]))
    assert np.isfinite(rmse) and rmse < 5.0
    assert 0 <= cov95 <= 1
    # ESS OK
    assert (res_A["ess"] > 0).all()
    assert (res_A["ess"] <= 50).all()
