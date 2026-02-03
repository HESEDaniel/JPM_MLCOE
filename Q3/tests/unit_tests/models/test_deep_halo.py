"""Unit tests for DeepHalo models."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.models.deep_halo import (
    DeepHaloLayer,
    FeatureBasedDeepHalo,
    FeaturelessDeepHalo,
)


# Fixtures

@pytest.fixture
def featureless_model():
    """Create a basic FeaturelessDeepHalo model."""
    model = FeaturelessDeepHalo(n_items=5, width=10, n_layers=2)
    model.instantiate()
    return model


@pytest.fixture
def feature_model():
    """Create a basic FeatureBasedDeepHalo model."""
    model = FeatureBasedDeepHalo(embedding_dim=16, n_layers=2, n_heads=2)
    model.instantiate(n_items_features=3)
    return model


@pytest.fixture
def deep_halo_layer():
    """Create a basic DeepHaloLayer."""
    layer = DeepHaloLayer(embedding_dim=16, n_heads=4)
    layer.build((None, None, 16))
    return layer


class TestFeaturelessDeepHalo:
    """Unit tests for FeaturelessDeepHalo."""

    # Instantiation Tests

    def test_instantiation(self, featureless_model):
        """Test model instantiation creates correct number of weights."""
        assert featureless_model.instantiated
        assert len(featureless_model.trainable_weights) == 3  # n_layers=2

    def test_weight_shapes(self, featureless_model):
        """Test weight matrices have correct shapes."""
        weights = featureless_model.trainable_weights
        assert weights[0].shape == (5, 10)   # (n_items, width)
        assert weights[-1].shape == (10, 5)  # (width, n_items)

    def test_single_layer_weights(self):
        """Test n_layers=1 creates only input and output projections."""
        model = FeaturelessDeepHalo(n_items=5, width=10, n_layers=1)
        model.instantiate()

        assert len(model.trainable_weights) == 2
        assert model.trainable_weights[0].shape == (5, 10)
        assert model.trainable_weights[1].shape == (10, 5)

    # Initialization Tests

    @pytest.mark.parametrize("init", ["normal", "he", "glorot", "He", "GLOROT"])
    def test_init_types(self, init):
        """Test initialization types work (case insensitive)."""
        model = FeaturelessDeepHalo(n_items=5, width=10, n_layers=2, init=init)
        model.instantiate()
        assert model.instantiated

    def test_invalid_init_raises(self):
        """Test invalid init raises ValueError."""
        model = FeaturelessDeepHalo(n_items=5, width=10, n_layers=2, init="invalid")
        with pytest.raises(ValueError, match="Unknown init"):
            model.instantiate()

    @pytest.mark.parametrize("init,limit", [
        ("he", np.sqrt(1.0 / 5)),      # PyTorch nn.Linear default: sqrt(1/fan_in)
        ("glorot", np.sqrt(6.0 / 15)), # TensorFlow GlorotUniform: sqrt(6/(fan_in+fan_out))
    ])
    def test_init_range(self, init, limit):
        """Test initialization values are in correct range."""
        model = FeaturelessDeepHalo(n_items=5, width=10, n_layers=2, init=init)
        model.instantiate()

        w = model.trainable_weights[0].numpy()
        assert w.max() <= limit
        assert w.min() >= -limit

    # Output Shape Tests

    @pytest.mark.parametrize("batch_size,n_items,width", [
        (8, 5, 10),   # standard
        (1, 5, 10),   # single batch
        (4, 1, 10),   # single item
        (4, 10, 5),   # width < n_items
    ])
    def test_output_shape(self, batch_size, n_items, width):
        """Test compute_batch_utility output shape."""
        model = FeaturelessDeepHalo(n_items=n_items, width=width, n_layers=2)
        model.instantiate()

        available = np.ones((batch_size, n_items), dtype=np.float32)
        output = model.compute_batch_utility(None, None, available, None)
        assert output.shape == (batch_size, n_items)

    # Computation Correctness Tests

    def test_quadratic_residual_computation(self):
        """Test y = y + theta @ y^2 computation with known weights."""
        model = FeaturelessDeepHalo(n_items=2, width=3, n_layers=2)
        model.instantiate()

        model._trainable_weights[0].assign([[1, 0, 0], [0, 1, 0]])
        model._trainable_weights[1].assign(np.eye(3))
        model._trainable_weights[2].assign([[1, 0], [0, 1], [0, 0]])

        available = np.array([[1, 0]], dtype=np.float32)
        output = model.compute_batch_utility(None, None, available, None)

        assert np.isclose(output[0, 0].numpy(), 2.0)
        assert output[0, 1].numpy() == tf.float32.min

    def test_unavailable_masked_to_neg_inf(self, featureless_model):
        """Test unavailable items get -inf, available items are finite."""
        available = np.array([[1, 1, 0, 0, 0]], dtype=np.float32)
        output = featureless_model.compute_batch_utility(None, None, available, None)

        assert output[0, 2].numpy() == tf.float32.min
        assert np.isfinite(output[0, 0].numpy())

    def test_halo_effect(self, featureless_model):
        """Test same item gets different utility with different context."""
        avail1 = np.array([[1, 1, 1, 0, 0]], dtype=np.float32)
        avail2 = np.array([[1, 0, 0, 1, 1]], dtype=np.float32)

        out1 = featureless_model.compute_batch_utility(None, None, avail1, None)
        out2 = featureless_model.compute_batch_utility(None, None, avail2, None)

        assert not np.isclose(out1[0, 0].numpy(), out2[0, 0].numpy())


class TestFeatureBasedDeepHalo:
    """Unit tests for FeatureBasedDeepHalo."""

    # Instantiation Tests

    def test_instantiation(self, feature_model):
        """Test model instantiation."""
        assert feature_model.instantiated
        assert len(feature_model.trainable_weights) > 0

    def test_chi_structure(self, feature_model):
        """Test chi is a 3-layer MLP with LayerNorm."""
        assert len(feature_model.chi.layers) == 4
        assert isinstance(feature_model.chi.layers[0], tf.keras.layers.Dense)
        assert isinstance(feature_model.chi.layers[1], tf.keras.layers.Dense)
        assert isinstance(feature_model.chi.layers[2], tf.keras.layers.Dense)
        assert isinstance(feature_model.chi.layers[3], tf.keras.layers.LayerNormalization)

    def test_single_layer_no_deephalo_layers(self):
        """Test n_layers=1 creates no DeepHaloLayers."""
        model = FeatureBasedDeepHalo(embedding_dim=16, n_layers=1, n_heads=2)
        model.instantiate(n_items_features=3)

        assert len(model.deep_halo_layers) == 0
        assert model.instantiated

    # Output Shape Tests

    @pytest.mark.parametrize("batch_size,n_items,n_features", [
        (8, 5, 3),    # standard
        (1, 5, 3),    # single batch
        (4, 1, 3),    # single item
        (4, 5, 1),    # single feature
    ])
    def test_output_shape(self, batch_size, n_items, n_features):
        """Test compute_batch_utility output shape."""
        model = FeatureBasedDeepHalo(embedding_dim=16, n_layers=2, n_heads=2)
        model.instantiate(n_items_features=n_features)

        features = np.random.randn(batch_size, n_items, n_features).astype(np.float32)
        available = np.ones((batch_size, n_items), dtype=np.float32)

        output = model.compute_batch_utility(None, features, available, None)
        assert output.shape == (batch_size, n_items)

    def test_tuple_features_input(self, feature_model):
        """Test tuple input for items_features_by_choice."""
        features1 = np.random.randn(4, 5, 2).astype(np.float32)
        features2 = np.random.randn(4, 5, 1).astype(np.float32)  # 2+1=3
        available = np.ones((4, 5), dtype=np.float32)

        output = feature_model.compute_batch_utility(None, (features1, features2), available, None)
        assert output.shape == (4, 5)

    # Error Handling Tests

    def test_zero_availability_raises(self, feature_model):
        """Test zero availability raises InvalidArgumentError."""
        features = np.random.randn(4, 5, 3).astype(np.float32)
        available = np.ones((4, 5), dtype=np.float32)
        available[2, :] = 0

        with pytest.raises(tf.errors.InvalidArgumentError):
            feature_model.compute_batch_utility(None, features, available, None)

    # Computation Correctness Tests

    def test_halo_effect(self, feature_model):
        """Test same item/features get different utility with different availability."""
        features = np.array([[[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]]], dtype=np.float32)

        avail1 = np.array([[1, 1, 1, 0]], dtype=np.float32)
        avail2 = np.array([[1, 0, 0, 1]], dtype=np.float32)

        out1 = feature_model.compute_batch_utility(None, features, avail1, None)
        out2 = feature_model.compute_batch_utility(None, features, avail2, None)

        assert not np.isclose(out1[0, 0].numpy(), out2[0, 0].numpy())

    def test_chi_transforms_features(self, feature_model):
        """Test chi MLP transforms features."""
        features1 = np.zeros((1, 2, 3), dtype=np.float32)
        features2 = np.ones((1, 2, 3), dtype=np.float32)
        available = np.ones((1, 2), dtype=np.float32)

        out1 = feature_model.compute_batch_utility(None, features1, available, None)
        out2 = feature_model.compute_batch_utility(None, features2, available, None)

        assert not np.allclose(out1.numpy(), out2.numpy())


class TestDeepHaloLayer:
    """Unit tests for DeepHaloLayer."""

    # Build Tests

    def test_layer_build(self, deep_halo_layer):
        """Test layer builds with correct attributes."""
        assert hasattr(deep_halo_layer, "W")
        assert hasattr(deep_halo_layer, "phi_fc1")
        assert hasattr(deep_halo_layer, "phi_fc2")
        assert hasattr(deep_halo_layer, "phi_norm")

    # Output Shape Tests

    @pytest.mark.parametrize("batch_size,n_items,n_heads", [
        (8, 5, 4),    # standard
        (1, 5, 4),    # single batch
        (4, 1, 4),    # single item
        (4, 5, 1),    # single head
    ])
    def test_output_shape(self, batch_size, n_items, n_heads):
        """Test output shape matches input shape."""
        layer = DeepHaloLayer(embedding_dim=16, n_heads=n_heads)
        layer.build((None, None, 16))

        z = tf.random.normal((batch_size, n_items, 16))
        available = tf.ones((batch_size, n_items))

        output = layer(z, z, available)
        assert output.shape == (batch_size, n_items, 16)

    # Masking Tests

    def test_masking(self, deep_halo_layer):
        """Test unavailable items are zero, available items are non-zero."""
        z = tf.random.normal((2, 5, 16))
        available = np.array([[1, 1, 0, 1, 1], [1, 0, 0, 0, 0]], dtype=np.float32)

        output = deep_halo_layer(z, z, available)

        # Unavailable items should be zero
        assert np.allclose(output[0, 2, :].numpy(), 0.0)
        assert np.allclose(output[1, 3, :].numpy(), 0.0)
        # Available items should be non-zero
        assert not np.allclose(output[0, 0, :].numpy(), 0.0)
        assert not np.allclose(output[1, 0, :].numpy(), 0.0)
