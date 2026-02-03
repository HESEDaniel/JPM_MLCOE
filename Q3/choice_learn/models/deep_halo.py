"""Implementation of DeepHalo models from Zhang et al. (2025)."""

import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel


class FeaturelessDeepHalo(ChoiceModel):
    """Featureless DeepHalo model."""

    def __init__(
        self,
        n_items,
        width=None,
        n_layers=3,
        init="he",
        **kwargs,
    ):
        """Initialize FeaturelessDeepHalo.

        Parameters
        ----------
        n_items : int
            Number of items J in the universe.
        width : int, optional
            Hidden dimension J'.
        n_layers : int, optional
            Number of layers L.
        init : str, optional
            Weight initialization: "normal", "glorot" (TensorFlow default), or "he" (PyTorch default).
        """
        super().__init__(**kwargs)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.n_items = n_items
        self.width = width or n_items
        self.n_layers = n_layers
        self.init = init
        self.instantiated = False
        self._trainable_weights = []

    def _get_initializer(self):
        """Get TensorFlow initializer based on init type."""
        init = self.init.lower()
        if init == "normal":
            return tf.keras.initializers.RandomNormal()
        elif init == "glorot":
            return tf.keras.initializers.GlorotUniform()
        elif init == "he":
            # PyTorch nn.Linear default: kaiming_uniform_ with a=sqrt(5)
            # bound = sqrt(1 / fan_in), NOT TensorFlow's HeUniform which uses sqrt(6 / fan_in)
            def pytorch_init(shape, dtype=tf.float32):
                bound = tf.sqrt(1.0 / shape[0])
                return tf.random.uniform(shape, -bound, bound, dtype=dtype)
            return pytorch_init
        else:
            raise ValueError(
                f"Unknown init: {self.init}. Use 'normal', 'glorot', or 'he'."
            )

    def instantiate(self):
        """Create weight matrices for the quadratic residual network."""
        self._trainable_weights = []
        initializer = self._get_initializer()

        # First Layer: J -> J'
        self._trainable_weights.append(
            tf.Variable(initializer([self.n_items, self.width]), dtype=tf.float32)
        )

        # Layer 2,...,n_layers: J' -> J'
        for _ in range(self.n_layers - 1):
            self._trainable_weights.append(
                tf.Variable(initializer([self.width, self.width]), dtype=tf.float32)
            )

        # Output projection: J' -> J
        self._trainable_weights.append(
            tf.Variable(initializer([self.width, self.n_items]), dtype=tf.float32)
        )

        self.instantiated = True

    @property
    def trainable_weights(self):
        """Return list of trainable weight tensors."""
        return self._trainable_weights

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute utilities via quadratic residual network.

        Parameters
        ----------
        shared_features_by_choice : tuple of np.ndarray
            Shared features (not used).
        items_features_by_choice : tuple of np.ndarray
            Item features (not used).
        available_items_by_choice : np.ndarray
            Availability mask of shape (n_choices, n_items).
        choices : np.ndarray
            Chosen items (not used).

        Returns
        -------
        tf.Tensor
            Utilities of shape (n_choices, n_items).
        """
        del shared_features_by_choice, items_features_by_choice, choices

        # y^0 = availability indicator (batch, J)
        y = tf.cast(available_items_by_choice, tf.float32)

        # Input projection: (batch, J) @ (J, J') -> (batch, J')
        y = tf.matmul(y, self._trainable_weights[0])

        # Hidden layers with quadratic residual: y = y + Theta @ y^2
        for theta in self._trainable_weights[1:-1]:
            y += tf.matmul(tf.square(y), theta)

        # Output projection: (batch, J') @ (J', J) -> (batch, J)
        utilities = tf.matmul(y, self._trainable_weights[-1])

        # Mask unavailable items to -inf for proper softmax normalization
        return tf.where(
            tf.cast(available_items_by_choice, tf.bool),
            utilities,
            tf.float32.min,
        )

    def fit(self, choice_dataset, **kwargs):
        """Fit the model, auto-instantiating if needed."""
        if not self.instantiated:
            self.instantiate()
        return super().fit(choice_dataset, **kwargs)


class DeepHaloLayer(tf.keras.layers.Layer):
    """Single DeepHalo layer for feature-based model."""

    def __init__(self, embedding_dim, n_heads, **kwargs):
        """Initialize DeepHaloLayer.

        Parameters
        ----------
        embedding_dim : int
            Dimension d of item embeddings.
        n_heads : int
            Number of interaction heads H.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

    def build(self, input_shape):
        """Build layer weights."""
        super().build(input_shape)
        d, H = self.embedding_dim, self.n_heads

        # Context projection: W^l (d -> H)
        self.W = self.add_weight(
            shape=(d, H), initializer="glorot_normal", name="context_projection"
        )

        # phi: fc1 expands to all heads, fc2 is shared
        self.phi_fc1 = tf.keras.layers.Dense(
            d * H, use_bias=True, activation="relu", name="phi_fc1"
        )
        self.phi_fc2 = tf.keras.layers.Dense(d, use_bias=True, name="phi_fc2")
        self.phi_norm = tf.keras.layers.LayerNormalization(name="phi_norm")

        # Build layers
        self.phi_fc1.build((None, d))
        self.phi_fc2.build((None, d))
        self.phi_norm.build((None, None, d))

    def call(self, z_prev, z_base, available_mask):
        """Forward pass of DeepHalo layer.

        Parameters
        ----------
        z_prev : tf.Tensor
            Previous layer embeddings of shape (batch, n_items, d).
        z_base : tf.Tensor
            Base embeddings of shape (batch, n_items, d).
        available_mask : tf.Tensor
            Availability mask of shape (batch, n_items).

        Returns
        -------
        tf.Tensor
            Updated embeddings of shape (batch, n_items, d).
        """
        batch_size = tf.shape(z_base)[0]
        n_items = tf.shape(z_base)[1]
        d, H = self.embedding_dim, self.n_heads

        # Mask: (batch, n_items, 1)
        mask = tf.expand_dims(tf.cast(available_mask, tf.float32), axis=-1)

        # Context summary: Z_bar = mean(W @ z_prev) over available items
        z_projected = tf.einsum("bid,dh->bih", z_prev, self.W) * mask
        n_available = tf.reduce_sum(mask, axis=1)
        z_bar = tf.reduce_sum(z_projected, axis=1) / n_available  # (batch, H)

        # phi transformation
        phi = self.phi_fc1(z_base)                              # (batch, n_items, d*H)
        phi = tf.reshape(phi, [batch_size, n_items, H, d])      # (batch, n_items, H, d)
        phi = self.phi_fc2(phi)                                 # (batch, n_items, H, d)
        phi = self.phi_norm(phi)                                # (batch, n_items, H, d)
        phi = phi * tf.expand_dims(mask, axis=2)                # Mask unavailable

        # Gating: Z_bar_h * phi_h, sum over heads
        z_bar_expanded = tf.reshape(z_bar, [batch_size, 1, H, 1])
        residuals = phi * z_bar_expanded
        residuals_avg = tf.reduce_sum(residuals, axis=2) / H

        # Residual and mask
        z_new = z_prev + residuals_avg
        z_new = z_new * mask

        return z_new


class FeatureBasedDeepHalo(ChoiceModel):
    """Feature-based DeepHalo with explicit interaction order control."""

    def __init__(
        self,
        embedding_dim=32,
        n_layers=3,
        n_heads=4,
        optimizer="adam",
        lr=1e-3,
        epochs=100,
        batch_size=32,
        **kwargs,
    ):
        """Initialize FeatureBasedDeepHalo.

        Parameters
        ----------
        embedding_dim : int, optional
            Dimension d of item embeddings.
        n_layers : int, optional
            Number of DeepHalo layers L.
        n_heads : int, optional
            Number of interaction heads H per layer.
        optimizer : str, optional
            Optimizer name.
        lr : float, optional
            Learning rate.
        epochs : int, optional
            Number of training epochs.
        batch_size : int, optional
            Batch size.
        """
        super().__init__(
            optimizer=optimizer,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs,
        )
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.instantiated = False

    def instantiate(self, n_items_features):
        """Build the model layers.

        Parameters
        ----------
        n_items_features : int
            Number of item features (input dimension for embedding MLP).
        """

        self.n_items_features = n_items_features
        d = self.embedding_dim

        # Base embedding MLP chi: 3-layer MLP with LayerNorm
        self.chi = tf.keras.Sequential([
            tf.keras.layers.Dense(d, use_bias=True, activation="relu"),
            tf.keras.layers.Dense(d, use_bias=True, activation="relu"),
            tf.keras.layers.Dense(d, use_bias=True, activation=None),
            tf.keras.layers.LayerNormalization(),
        ], name="chi_embedding")
        self.chi.build((None, n_items_features))

        # DeepHalo layers
        self.deep_halo_layers = []
        for layer_idx in range(self.n_layers - 1):
            layer = DeepHaloLayer(
                embedding_dim=self.embedding_dim,
                n_heads=self.n_heads,
                name=f"deep_halo_layer_{layer_idx}",
            )
            layer.build((None, None, self.embedding_dim))
            self.deep_halo_layers.append(layer)

        # Final projection beta
        self.beta = tf.keras.layers.Dense(1, use_bias=False, name="utility_projection")
        self.beta.build((None, self.embedding_dim))

        self.instantiated = True

    @property
    def trainable_weights(self):
        """Return list of trainable weight tensors."""
        weights = []
        weights.extend(self.chi.trainable_weights)
        for layer in self.deep_halo_layers:
            weights.extend(layer.trainable_weights)
        weights.extend(self.beta.trainable_weights)
        return weights

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute utilities using feature-based DeepHalo.

        Parameters
        ----------
        shared_features_by_choice : tuple of np.ndarray
            Shared features (not used).
        items_features_by_choice : tuple of np.ndarray
            Item features of shape (n_choices, n_items, n_features).
        available_items_by_choice : np.ndarray
            Availability mask of shape (n_choices, n_items).
        choices : np.ndarray
            Chosen items (not used).

        Returns
        -------
        tf.Tensor
            Utilities of shape (n_choices, n_items).
        """
        _ = shared_features_by_choice, choices

        # Validate: each choice must have at least one available item
        n_available = tf.reduce_sum(tf.cast(available_items_by_choice, tf.float32), axis=1)
        tf.debugging.assert_positive(
            n_available,
            message="Each choice must have at least one available item",
        )

        # Handle tuple input
        if isinstance(items_features_by_choice, tuple):
            items_features_by_choice = tf.concat(
                [tf.cast(f, tf.float32) for f in items_features_by_choice],
                axis=-1,
            )
        items_features = tf.cast(items_features_by_choice, tf.float32)

        # Get shapes
        batch_size = tf.shape(items_features)[0]
        n_items = tf.shape(items_features)[1]

        # Reshape for MLP: (batch * n_items, n_features)
        features_flat = tf.reshape(items_features, [-1, self.n_items_features])

        # Apply base embedding chi
        z_base_flat = self.chi(features_flat)

        # Reshape back: (batch, n_items, embedding_dim)
        z_base = tf.reshape(z_base_flat, [batch_size, n_items, self.embedding_dim])

        # Apply DeepHalo layers
        z = z_base
        for layer in self.deep_halo_layers:
            z = layer(z, z_base, available_items_by_choice)

        # Apply final projection beta
        return tf.squeeze(self.beta(z), axis=-1)

    def fit(
        self,
        choice_dataset,
        sample_weight=None,
        val_dataset=None,
        validation_freq=1,
        verbose=0,
    ):
        """Fit the model to a ChoiceDataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Training dataset.
        sample_weight : np.ndarray, optional
            Sample weights.
        val_dataset : ChoiceDataset, optional
            Validation dataset.
        validation_freq : int, optional
            Validation frequency in epochs. Default is 1.
        verbose : int, optional
            Verbosity level. Default is 0.

        Returns
        -------
        dict
            Training history with loss values.
        """
        if not self.instantiated:
            items_features = choice_dataset.items_features_by_choice

            if items_features is not None:
                if isinstance(items_features, tuple):
                    n_items_features = sum(f.shape[-1] for f in items_features)
                else:
                    n_items_features = items_features.shape[-1]
            else:
                raise ValueError("FeatureBasedDeepHalo requires items_features_by_choice")

            self.instantiate(n_items_features)

        return super().fit(
            choice_dataset,
            sample_weight=sample_weight,
            val_dataset=val_dataset,
            validation_freq=validation_freq,
            verbose=verbose,
        )
