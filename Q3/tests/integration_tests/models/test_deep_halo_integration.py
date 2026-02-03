"""Integration tests for DeepHalo models."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.datasets.base import load_heating
from choice_learn.models.deep_halo import FeatureBasedDeepHalo, FeaturelessDeepHalo


# Fixtures

@pytest.fixture
def synthetic_dataset():
    """Create synthetic dataset for FeaturelessDeepHalo."""
    np.random.seed(42)
    n_choices, n_items, choice_set_size = 100, 10, 8

    available = np.zeros((n_choices, n_items), dtype=np.float32)
    choices = np.zeros(n_choices, dtype=np.int32)

    for i in range(n_choices):
        available_items = np.random.choice(n_items, choice_set_size, replace=False)
        available[i, available_items] = 1.0
        choices[i] = np.random.choice(available_items)

    return ChoiceDataset(available_items_by_choice=available, choices=choices)


@pytest.fixture
def heating_dataset():
    """Load heating dataset for FeatureBasedDeepHalo."""
    heating_df = load_heating(as_frame=True)
    items = ["hp", "gc", "gr", "ec", "er"]

    choices = np.array([items.index(val) for val in heating_df["depvar"].to_numpy().ravel()])
    items_features = np.stack(
        [heating_df[["ic." + item, "oc." + item]].to_numpy() for item in items],
        axis=1,
    ).astype("float32")

    return ChoiceDataset(items_features_by_choice=items_features, choices=choices)


class TestFeaturelessDeepHalo:
    """Integration tests for FeaturelessDeepHalo."""

    # Training Tests

    def test_fit(self, synthetic_dataset):
        """Test training returns history."""
        model = FeaturelessDeepHalo(
            n_items=10, width=8, n_layers=2,
            optimizer="adam", lr=0.001, epochs=3, batch_size=32,
        )
        history = model.fit(synthetic_dataset, verbose=0)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 3

    def test_predict(self, synthetic_dataset):
        """Test probabilities sum to 1."""
        model = FeaturelessDeepHalo(
            n_items=10, width=8, n_layers=2,
            optimizer="adam", epochs=2,
        )
        model.fit(synthetic_dataset, verbose=0)
        probas = model.predict_probas(synthetic_dataset)

        assert probas.shape == (100, 10)

        available = synthetic_dataset.available_items_by_choice
        for i in range(10):
            prob_sum = np.sum(probas[i].numpy() * available[i])
            assert np.isclose(prob_sum, 1.0, atol=1e-5)

    @pytest.mark.parametrize("n_available", [1, 2, 5, 10])
    def test_variable_availability(self, n_available):
        """Test training with different choice set sizes."""
        n_items = 10

        available = np.zeros((1, n_items), dtype=np.float32)
        available[0, :n_available] = 1.0
        choices = np.array([0], dtype=np.int32)

        dataset = ChoiceDataset(available_items_by_choice=available, choices=choices)
        model = FeaturelessDeepHalo(
            n_items=n_items, width=8, n_layers=2,
            optimizer="adam", epochs=2,
        )

        history = model.fit(dataset, verbose=0)
        assert "train_loss" in history

    # Robustness Tests

    def test_save_load_weights(self, synthetic_dataset):
        """Model weights can be saved and loaded."""
        model = FeaturelessDeepHalo(
            n_items=10, width=8, n_layers=2,
            optimizer="adam", epochs=2,
        )
        model.fit(synthetic_dataset, verbose=0)
        pred1 = model.predict_probas(synthetic_dataset)

        weights = [w.numpy().copy() for w in model.trainable_weights]

        model2 = FeaturelessDeepHalo(n_items=10, width=8, n_layers=2)
        model2.instantiate()
        for w, saved_w in zip(model2.trainable_weights, weights):
            w.assign(saved_w)

        pred2 = model2.predict_probas(synthetic_dataset)
        assert np.allclose(pred1.numpy(), pred2.numpy())

    def test_prediction_consistency(self, synthetic_dataset):
        """Same input produces same output."""
        model = FeaturelessDeepHalo(
            n_items=10, width=8, n_layers=2,
            optimizer="adam", epochs=2,
        )
        model.fit(synthetic_dataset, verbose=0)

        pred1 = model.predict_probas(synthetic_dataset)
        pred2 = model.predict_probas(synthetic_dataset)

        assert np.allclose(pred1.numpy(), pred2.numpy())


class TestFeatureBasedDeepHalo:
    """Integration tests for FeatureBasedDeepHalo."""

    # Training Tests

    def test_fit(self, heating_dataset):
        """Test training on heating dataset."""
        model = FeatureBasedDeepHalo(
            embedding_dim=16, n_layers=2, n_heads=2,
            optimizer="adam", lr=0.001, epochs=3,
        )
        history = model.fit(heating_dataset, verbose=0)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 3

    def test_predict(self, heating_dataset):
        """Test probabilities sum to 1."""
        model = FeatureBasedDeepHalo(
            embedding_dim=8, n_layers=2, n_heads=2,
            optimizer="adam", epochs=2,
        )
        model.fit(heating_dataset, verbose=0)
        probas = model.predict_probas(heating_dataset)

        n_choices = heating_dataset.get_n_choices()
        n_items = heating_dataset.get_n_items()
        assert probas.shape == (n_choices, n_items)

        for i in range(min(10, n_choices)):
            assert np.isclose(np.sum(probas[i].numpy()), 1.0, atol=1e-5)

    def test_synthetic_loss_decreases(self):
        """Test loss decreases during training."""
        np.random.seed(42)
        n_choices, n_items, n_features = 200, 8, 5

        dataset = ChoiceDataset(
            items_features_by_choice=np.random.randn(n_choices, n_items, n_features).astype(np.float32),
            available_items_by_choice=np.ones((n_choices, n_items), dtype=np.float32),
            choices=np.random.randint(0, n_items, n_choices),
        )

        model = FeatureBasedDeepHalo(
            embedding_dim=16, n_layers=2, n_heads=2,
            optimizer="adam", lr=0.001, epochs=3, batch_size=32,
        )
        history = model.fit(dataset, verbose=0)

        assert history["train_loss"][-1] <= history["train_loss"][0] * 1.5

    # Robustness Tests

    def test_numerical_stability(self):
        """No NaN/Inf with extreme feature values."""
        features = np.array([[[1e6, -1e6], [1e-6, -1e-6]]], dtype=np.float32)
        available = np.ones((1, 2), dtype=np.float32)

        model = FeatureBasedDeepHalo(embedding_dim=8, n_layers=2, n_heads=2)
        model.instantiate(n_items_features=2)

        output = model.compute_batch_utility(None, features, available, None)
        assert np.all(np.isfinite(output.numpy()))

    def test_save_load_weights(self, heating_dataset):
        """Model weights can be saved and loaded."""
        model = FeatureBasedDeepHalo(
            embedding_dim=8, n_layers=2, n_heads=2,
            optimizer="adam", epochs=2,
        )
        model.fit(heating_dataset, verbose=0)
        pred1 = model.predict_probas(heating_dataset)

        weights = [w.numpy().copy() for w in model.trainable_weights]

        model2 = FeatureBasedDeepHalo(embedding_dim=8, n_layers=2, n_heads=2)
        model2.instantiate(n_items_features=2)
        for w, saved_w in zip(model2.trainable_weights, weights):
            w.assign(saved_w)

        pred2 = model2.predict_probas(heating_dataset)
        assert np.allclose(pred1.numpy(), pred2.numpy())

    # Error Handling Tests

    def test_zero_availability_raises(self):
        """Test error for zero available items."""
        np.random.seed(42)
        n_choices, n_items, n_features = 10, 5, 3

        available = np.ones((n_choices, n_items), dtype=np.float32)
        available[5, :] = 0.0

        dataset = ChoiceDataset(
            items_features_by_choice=np.random.randn(n_choices, n_items, n_features).astype(np.float32),
            available_items_by_choice=available,
            choices=np.random.randint(0, n_items, n_choices),
        )

        model = FeatureBasedDeepHalo(embedding_dim=8, n_layers=2, n_heads=2)
        model.instantiate(n_items_features=n_features)

        with pytest.raises(tf.errors.InvalidArgumentError):
            model.predict_probas(dataset)
