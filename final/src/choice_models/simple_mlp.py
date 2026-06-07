"""SimpleMLP baseline: V_j = MLP(slate_indicator)_j + psi_j * x_t.

A plain neural baseline whose input is the slate indicator (B, M). Used to
quantify whether DeepHalo's specific architecture (the quadratic residual
block) adds value over a vanilla MLP on the same input.
"""

from __future__ import annotations

import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel


class SimpleMLP(ChoiceModel):
    """Plain MLP over the slate indicator -> per-offer logits, plus a macro shift."""

    def __init__(
        self,
        M: int,
        hidden: int = 64,
        n_layers: int = 2,
        use_macro: bool = True,
        optimizer: str = "adam",
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 512,
        loss_type: str = "nll",
        **kwargs,
    ):
        """Build the MLP baseline and optional macro-shift parameter.

        Parameters
        ----------
        M : int
            Number of offers (per-offer logit dimension and MLP input width).
        hidden : int
            Width of each hidden Dense layer.
        n_layers : int
            Number of ReLU hidden layers before the final per-offer head.
        use_macro : bool
            If True, add a learnable per-offer macro shift ``psi_offer * x_t``.
        optimizer : str
            Optimizer name passed to the base ChoiceModel.
        lr : float
            Learning rate.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        loss_type : str
            "nll" for the default negative-log-likelihood loss, or "mse" to
            use mean-squared error instead.
        **kwargs
            Forwarded to the base ChoiceModel.
        """
        super().__init__(
            optimizer=optimizer, lr=lr, epochs=epochs, batch_size=batch_size, **kwargs
        )
        if loss_type == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        self.M = M
        self.use_macro = use_macro

        layers = []
        for _ in range(n_layers):
            layers.append(tf.keras.layers.Dense(hidden, activation="relu"))
        layers.append(tf.keras.layers.Dense(M, activation=None))
        self.mlp = tf.keras.Sequential(layers, name="simple_mlp")
        self.mlp.build((None, M))

        if use_macro:
            self.psi_offer = tf.Variable(
                tf.zeros([M], dtype=tf.float32), name="psi_offer", trainable=True
            )
        else:
            self.psi_offer = None
        self.instantiated = True

    @property
    def trainable_weights(self):
        """MLP weights plus the macro-shift parameter when macro is enabled."""
        ws = list(self.mlp.trainable_weights)
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
        """Compute per-offer utilities V_j = MLP(slate_indicator)_j + psi_j * x_t.

        Parameters
        ----------
        shared_features_by_choice : tf.Tensor or tuple or None
            Per-choice shared (macro) features. The first column is taken as
            the macro driver x_t for the macro shift. Tuples are concatenated
            along the last axis. Ignored when macro is disabled.
        items_features_by_choice : tf.Tensor or tuple
            Per-offer features (unused by this baseline; deleted).
        available_items_by_choice : tf.Tensor
            Slate indicator of shape (B, M) marking which offers are available;
            this is the MLP input.
        choices : tf.Tensor
            Observed choices. Unused here (deleted) since utilities do not
            depend on the realized choice.

        Returns
        -------
        tf.Tensor
            Per-offer utilities of shape (B, M).
        """
        del items_features_by_choice, choices
        mlp_input = tf.cast(available_items_by_choice, tf.float32)              # (B, M)
        h = self.mlp(mlp_input)                                                # (B, M)

        if self.use_macro and shared_features_by_choice is not None:
            if isinstance(shared_features_by_choice, tuple):
                shared_features_by_choice = tf.concat(
                    [tf.cast(f, tf.float32) for f in shared_features_by_choice], axis=-1
                )
            x_t = tf.cast(shared_features_by_choice[:, 0], tf.float32)
            h = h + self.psi_offer[None, :] * x_t[:, None]
        return h
