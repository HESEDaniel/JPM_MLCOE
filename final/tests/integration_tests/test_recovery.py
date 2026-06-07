"""Integration tests - short fits must recover psi_j signal on DGP-B."""

import numpy as np
import tensorflow as tf

from src.datasets.dgp_macro import (
    DGPConfig, data_to_choice_dataset, simulate_macro_choice_dgp,
)
from src.choice_models.macro_deephalo import MacroDeepHalo
from src.choice_models.simple_mlp import SimpleMLP
from src.choice_models.simple_mnl import SimpleMNL


def _short_data():
    return simulate_macro_choice_dgp(
        scenario="B", cfg=DGPConfig(T=80, N_t=30), seed=0
    )


def test_featureless_recovery_short():
    tf.random.set_seed(0)
    data = _short_data()
    ds = data_to_choice_dataset(data)
    model = MacroDeepHalo(M=data["M"], epochs=15,
                          batch_size=256, optimizer="adam", lr=1e-3)
    model.fit(ds, verbose=0)
    pearson = np.corrcoef(data["_psi_offer"], model.psi_offer.numpy())[0, 1]
    assert pearson > 0.5, f"Featureless psi Pearson {pearson:.3f} should be > 0.5"


def test_simple_mnl_recovery_short():
    tf.random.set_seed(0)
    data = _short_data()
    ds = data_to_choice_dataset(data)
    model = SimpleMNL(M=data["M"], epochs=15, batch_size=256, optimizer="adam", lr=1e-2)
    model.fit(ds, verbose=0)
    pearson = np.corrcoef(data["_psi_offer"], model.psi_offer.numpy())[0, 1]
    assert pearson > 0.5, f"SimpleMNL psi Pearson {pearson:.3f} should be > 0.5"


def test_simple_mlp_recovery_short():
    tf.random.set_seed(0)
    data = _short_data()
    ds = data_to_choice_dataset(data)
    model = SimpleMLP(M=data["M"], epochs=15,
                      batch_size=256, optimizer="adam", lr=1e-3)
    model.fit(ds, verbose=0)
    pearson = np.corrcoef(data["_psi_offer"], model.psi_offer.numpy())[0, 1]
    assert pearson > 0.5, f"SimpleMLP psi Pearson {pearson:.3f} should be > 0.5"
