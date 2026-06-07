"""DeepHalo with offer-indexed macro loading, as a single ChoiceModel.

V_j(S, x_t) = h_DeepHalo(...)_j + psi_j * x_t

Wraps a FeaturelessDeepHalo (the offer category is hidden from the network).
Trained via standard choice-learn ``model.fit(choice_dataset)``. ``x_t`` enters
through ``shared_features_by_choice`` (the single shared scalar per choice).
"""

from __future__ import annotations

import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel

from src.choice_models.deephalo import FeaturelessDeepHalo


class MacroDeepHalo(ChoiceModel):
    """ChoiceModel pairing a FeaturelessDeepHalo NN with a per-offer macro loading psi_j."""

    def __init__(
        self,
        M: int,
        # DeepHalo NN architecture
        width: int | None = None,
        n_layers: int = 3,
        init: str = "he",
        # ChoiceModel training
        optimizer: str = "adam",
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 512,
        loss_type: str = "nll",  # "nll" (default) | "mse" (paper-style)
        # Initial psi_offer values. Default = zeros. For sign-flip identification
        # via domain-knowledge psi prior, pass an explicit ``psi_init`` vector of
        # shape (M,) -- e.g. ``sign(psi_true) * 0.5`` in synthetic experiments
        # (analog: economist's prior on each offer's cyclical character).
        psi_init=None,
        **kwargs,
    ):
        """Build a MacroDeepHalo and instantiate its inner DeepHalo and psi_offer.

        Parameters
        ----------
        M : int
            Number of offers (items).
        width : int or None
            Hidden width of the FeaturelessDeepHalo NN.
        n_layers : int
            Number of layers in the inner DeepHalo NN.
        init : str
            Weight initialization scheme for the FeaturelessDeepHalo NN.
        optimizer : str
            ChoiceModel optimizer name.
        lr : float
            Learning rate.
        epochs : int
            Number of training epochs.
        batch_size : int
            Training batch size.
        loss_type : str
            "nll" (default) or "mse" (paper-style). "mse" swaps in a
            MeanSquaredError loss.
        psi_init : array-like or None
            Initial psi_offer values of shape (M,). Default = zeros. For
            sign-flip identification via a domain-knowledge psi prior, pass an
            explicit vector, e.g. sign(psi_true) * 0.5 in synthetic experiments
            (analog: economist's prior on each offer's cyclical character).
        **kwargs
            Forwarded to the ChoiceModel base class.
        """
        super().__init__(
            optimizer=optimizer, lr=lr, epochs=epochs, batch_size=batch_size, **kwargs
        )
        if loss_type == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        self.M = M

        self.deephalo = FeaturelessDeepHalo(
            n_items=M, width=width, n_layers=n_layers, init=init,
        )
        self.deephalo.instantiate()

        if psi_init is None:
            psi_init_arr = tf.zeros([M], dtype=tf.float32)
        else:
            psi_init_arr = tf.cast(tf.convert_to_tensor(psi_init), tf.float32)
            assert psi_init_arr.shape == (M,), (
                f"psi_init must be shape ({M},), got {tuple(psi_init_arr.shape)}"
            )
        self.psi_offer = tf.Variable(psi_init_arr, name="psi_offer", trainable=True)
        self.instantiated = True

    @property
    def trainable_weights(self):
        """Inner DeepHalo weights plus the per-offer macro loading psi_offer."""
        return list(self.deephalo.trainable_weights) + [self.psi_offer]

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Per-offer utility V_j(S, x_t) = h_DeepHalo(...)_j + psi_j * x_t.

        x_t is taken from shared_features_by_choice[:, 0] (single scalar per
        choice).

        Parameters
        ----------
        shared_features_by_choice : tensor, tuple of tensors, or None
            Shared features per choice; x_t is the first scalar column. choice-
            learn's ChoiceDataset may pass this as a single array or a tuple of
            arrays (when multiple shared-feature groups exist). If None, only the
            inner DeepHalo utility is returned.
        items_features_by_choice : tensor
            Per-item features; ignored by the FeaturelessDeepHalo.
        available_items_by_choice : tensor
            Availability mask per item per choice.
        choices : tensor
            Observed choices (unused here; the inner NN is called with None).

        Returns
        -------
        tensor
            Utilities of shape (B, M).
        """
        # Inner FeaturelessDeepHalo NN (items_features ignored).
        h = self.deephalo.compute_batch_utility(
            shared_features_by_choice=None,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=None,
        )  # (B, M)

        if shared_features_by_choice is None:
            return h
        # choice-learn's ChoiceDataset may pass shared_features as either a single
        # array or a tuple of arrays (when multiple shared-feature groups exist).
        # Concatenate to a single tensor; x_t is always the first scalar column.
        if isinstance(shared_features_by_choice, tuple):
            shared_features_by_choice = tf.concat(
                [tf.cast(f, tf.float32) for f in shared_features_by_choice], axis=-1
            )
        x_t = tf.cast(shared_features_by_choice[:, 0], tf.float32)            # (B,)
        macro = self.psi_offer[None, :] * x_t[:, None]                        # (B, M)
        return h + macro
