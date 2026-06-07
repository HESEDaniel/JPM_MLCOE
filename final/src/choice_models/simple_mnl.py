"""SimpleMNL baseline: V_j = base_j + psi_j * x_t  (no halo, no NN).

Trained via standard choice-learn `model.fit(choice_dataset)`. Quantifies the
value of DeepHalo's halo-learning relative to a featureless MNL.
"""

from __future__ import annotations

import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel


class SimpleMNL(ChoiceModel):
    """Per-offer scalar utility + per-offer macro loading."""

    def __init__(
        self,
        M: int,
        use_macro: bool = True,
        optimizer: str = "adam",
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 512,
        loss_type: str = "nll",
        **kwargs,
    ):
        """Build the per-offer base utility and optional macro loading.

        Parameters
        ----------
        M : int
            Number of offers (alternatives).
        use_macro : bool
            If True, learn a per-offer macro loading psi_offer applied to the
            shared macro feature x_t; if False, the model reduces to per-offer
            intercepts only.
        optimizer : str
            Name of the optimizer passed to the choice-learn base model.
        lr : float
            Learning rate.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        loss_type : str
            Loss selector. "mse" switches the loss to mean squared error;
            any other value (e.g. "nll") keeps the base model's default.
        **kwargs
            Additional keyword arguments forwarded to the base model.
        """
        super().__init__(
            optimizer=optimizer, lr=lr, epochs=epochs, batch_size=batch_size, **kwargs
        )
        if loss_type == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        self.M = M
        self.use_macro = use_macro
        self.base = tf.Variable(
            tf.zeros([M], dtype=tf.float32), name="base", trainable=True
        )
        if use_macro:
            self.psi_offer = tf.Variable(
                tf.zeros([M], dtype=tf.float32), name="psi_offer", trainable=True
            )
        else:
            self.psi_offer = None
        self.instantiated = True

    @property
    def trainable_weights(self):
        """Trainable variables: base intercepts plus macro loading if enabled."""
        ws = [self.base]
        if self.use_macro:
            ws.append(self.psi_offer)
        return ws

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute per-offer utilities V_j = base_j + psi_j * x_t for a batch.

        Item-level features and the chosen index are unused (the model is
        featureless apart from the single shared macro signal x_t), so they are
        dropped. When the shared features arrive as a tuple they are cast to
        float32 and concatenated before the macro signal x_t is read from the
        first column.

        Parameters
        ----------
        shared_features_by_choice : tf.Tensor or tuple of tf.Tensor or None
            Shared (choice-level) features; the macro signal x_t is taken from
            the first column. None or use_macro=False leaves only the base term.
        items_features_by_choice : Any
            Item-level features (unused; dropped).
        available_items_by_choice : tf.Tensor
            Availability tensor of shape (B, M); only its leading dimension B is
            used to broadcast the base utilities.
        choices : Any
            Chosen item indices (unused; dropped).

        Returns
        -------
        tf.Tensor
            Utilities of shape (B, M).
        """
        del items_features_by_choice, choices
        B = tf.shape(available_items_by_choice)[0]
        utility = tf.broadcast_to(self.base[None, :], [B, self.M])
        if self.use_macro and shared_features_by_choice is not None:
            if isinstance(shared_features_by_choice, tuple):
                shared_features_by_choice = tf.concat(
                    [tf.cast(f, tf.float32) for f in shared_features_by_choice], axis=-1
                )
            x_t = tf.cast(shared_features_by_choice[:, 0], tf.float32)
            utility = utility + self.psi_offer[None, :] * x_t[:, None]
        return utility
