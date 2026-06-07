"""Unit tests for the choice models (MacroDeepHalo, SimpleMNL, SimpleMLP).

Tests:
  - compute_batch_utility output shape, finite
  - trainable_weights includes psi_offer
  - model.fit runs to completion on a small ChoiceDataset
"""

import numpy as np
import pytest
import tensorflow as tf

from src.datasets.dgp_macro import (
    DGPConfig, data_to_choice_dataset, simulate_macro_choice_dgp,
)
from src.choice_models.macro_deephalo import MacroDeepHalo
from src.choice_models.simple_mlp import SimpleMLP
from src.choice_models.simple_mnl import SimpleMNL


@pytest.fixture(scope="module")
def tiny_data():
    return simulate_macro_choice_dgp(
        scenario="B", cfg=DGPConfig(M=8, K_min=2, K_max=4, T=20, N_t=5), seed=0
    )


@pytest.fixture(scope="module")
def tiny_ds(tiny_data):
    return data_to_choice_dataset(tiny_data)


# ---------- MacroDeepHalo ----------


def test_macro_dh_utility_shape(tiny_data, tiny_ds):
    M, B = tiny_data["M"], tiny_data["T"] * tiny_data["N_t"]
    model = MacroDeepHalo(M=M, epochs=2, batch_size=64)
    u = model.compute_batch_utility(
        tiny_ds.shared_features_by_choice,
        tiny_ds.items_features_by_choice,
        tiny_ds.available_items_by_choice,
        tiny_ds.choices,
    )
    assert u.shape == (B, M)
    assert tf.reduce_all(tf.math.is_finite(u))


def test_macro_dh_trainable_weights_include_psi(tiny_data):
    M = tiny_data["M"]
    model = MacroDeepHalo(M=M)
    assert any(w is model.psi_offer for w in model.trainable_weights)


def test_macro_dh_fit_runs(tiny_data, tiny_ds):
    M = tiny_data["M"]
    model = MacroDeepHalo(M=M, epochs=3, batch_size=64, optimizer="adam", lr=1e-3)
    history = model.fit(tiny_ds, verbose=0)
    assert "train_loss" in history
    assert len(history["train_loss"]) == 3


# ---------- SimpleMNL ----------


def test_simple_mnl_utility_shape(tiny_data, tiny_ds):
    M, B = tiny_data["M"], tiny_data["T"] * tiny_data["N_t"]
    model = SimpleMNL(M=M, epochs=2)
    u = model.compute_batch_utility(
        tiny_ds.shared_features_by_choice,
        tiny_ds.items_features_by_choice,
        tiny_ds.available_items_by_choice,
        tiny_ds.choices,
    )
    assert u.shape == (B, M)


def test_simple_mnl_no_macro_omits_psi(tiny_data):
    model = SimpleMNL(M=tiny_data["M"], use_macro=False)
    assert model.psi_offer is None
    assert all(w is not None for w in model.trainable_weights)


def test_simple_mnl_fit_runs(tiny_data, tiny_ds):
    M = tiny_data["M"]
    model = SimpleMNL(M=M, epochs=3, batch_size=64, optimizer="adam", lr=1e-3)
    history = model.fit(tiny_ds, verbose=0)
    assert "train_loss" in history and len(history["train_loss"]) == 3


# ---------- SimpleMLP ----------


def test_simple_mlp_utility_shape(tiny_data, tiny_ds):
    M, B = tiny_data["M"], tiny_data["T"] * tiny_data["N_t"]
    model = SimpleMLP(M=M, epochs=2)
    u = model.compute_batch_utility(
        tiny_ds.shared_features_by_choice,
        tiny_ds.items_features_by_choice,
        tiny_ds.available_items_by_choice,
        tiny_ds.choices,
    )
    assert u.shape == (B, M)


def test_simple_mlp_fit_runs(tiny_data, tiny_ds):
    M = tiny_data["M"]
    model = SimpleMLP(M=M, epochs=3, batch_size=64, optimizer="adam", lr=1e-3)
    history = model.fit(tiny_ds, verbose=0)
    assert "train_loss" in history
