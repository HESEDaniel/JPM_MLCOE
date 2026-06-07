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
        # MSE = the paper's training loss. Effective only when this class is fitted
        # standalone (paper Q3-1b reproduction). When wrapped by MacroDeepHalo,
        # the outer's self.loss takes precedence and this attribute is unused.
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

        return utilities

    def fit(self, choice_dataset, **kwargs):
        """Fit the model, auto-instantiating if needed."""
        if not self.instantiated:
            self.instantiate()
        return super().fit(choice_dataset, **kwargs)
