"""Adapter exposing the dynamic Deep-Halo SSM through the Q2 SSM interface.

Q2 filters require ``m0``, ``P0``, ``f_batch``, ``Q_sampler``, ``log_likelihood``.
``y_t`` is encoded as the time index ``t`` and the actual slate/choice
tensors are looked up inside ``log_likelihood`` --- this lets the Q2 PF
loop iterate ``ys[t] = t`` without needing to know our observation
structure.
"""

from __future__ import annotations

import tensorflow as tf

from .q2.filters.common import DTYPE, masked_log_softmax


class DeepHaloMacroSSM:
    """Scalar AR(1) latent + DeepHalo+macro Categorical observation.

    Implements the Q2 SSM interface so that ``q2.filters.ParticleFilter``
    and ``q2.filters.DifferentiableParticleFilter`` (with any
    ``q2.resampling`` resampler) can be applied directly to our model.

    Parameters
    ----------
    choice_model : MacroDeepHalo
        Stage-2 trained model providing ``deephalo`` (frozen NN) and
        ``psi_offer`` (per-offer macro loading).
    data : dict
        Output of ``simulate_macro_choice_dgp``.  Used for the slate
        indicator and the choice tensor.
    mu, phi, sigma : float
        State-equation parameters (scalar AR(1)).
    """

    state_dim = 1

    def __init__(self, choice_model, data: dict,
                 # Filter mode: pass concrete values (Tensors or floats).
                 mu=None, phi=None, sigma=None, psi=None,
                 # Joint-MLE mode: pass *raw* Variables; the SSM applies
                 # the unconstrained -> constrained transform (tanh / exp)
                 # inside its forward pass so a single instance can be
                 # reused across Adam steps without triggering retracing.
                 mu_var=None, phi_raw_var=None, log_sigma_var=None,
                 psi_var=None,
                 precompute_h: bool = True):
        """Wrap our SSM in the Q2 interface.

        Two modes share this class. In filter mode (default) pass concrete
        ``mu, phi, sigma, psi`` and the properties return the cast values
        directly. In joint-MLE mode pass ``mu_var, phi_raw_var,
        log_sigma_var, psi_var``; the properties recompute the constrained
        values from the raw Variables on every access so a single SSM
        instance can be shared by every Adam step (this avoids the per-step
        ``tf.function`` retracing observed previously). Gradients flow
        through the unconstrained -> constrained transform (tanh / exp) back
        to the raw Variables.

        Parameters
        ----------
        choice_model : MacroDeepHalo
            Stage-2 trained model providing ``deephalo`` (frozen NN) and
            ``psi_offer`` (per-offer macro loading).
        data : dict
            Output of ``simulate_macro_choice_dgp``; supplies the slate
            indicator and the choice tensor.
        mu, phi, sigma : float or tf.Tensor, optional
            Concrete state-equation parameters (scalar AR(1)) for filter mode.
        psi : tf.Tensor, optional
            Concrete per-offer macro loading for filter mode; defaults to
            ``choice_model.psi_offer``.
        mu_var, phi_raw_var, log_sigma_var : tf.Variable, optional
            Raw (unconstrained) Variables for joint-MLE mode. Passing
            ``mu_var`` switches the instance into raw-vars mode.
        psi_var : tf.Variable, optional
            Raw per-offer macro loading for joint-MLE mode; defaults to
            ``choice_model.psi_offer``.
        precompute_h : bool, default True
            If True, cache the slate-only DeepHalo logits once at
            construction (filter mode); otherwise recompute them per step.
        """
        self.choice_model = choice_model
        self.slate = tf.constant(data["slate_indicator"], DTYPE)   # (T, N_t, M)
        self.choice = tf.constant(data["choice"], tf.int32)        # (T, N_t)
        self.T = int(data["T"])

        self._raw_vars_mode = mu_var is not None
        if self._raw_vars_mode:
            assert phi_raw_var is not None and log_sigma_var is not None, (
                "joint-MLE mode requires mu_var, phi_raw_var, log_sigma_var"
            )
            self.mu_var = mu_var
            self.phi_raw_var = phi_raw_var
            self.log_sigma_var = log_sigma_var
            self.psi_var = (psi_var if psi_var is not None
                            else choice_model.psi_offer)
        else:
            self._mu_value = tf.cast(mu, DTYPE)
            self._phi_value = tf.cast(phi, DTYPE)
            self._sigma_value = tf.cast(sigma, DTYPE)
            psi_source = psi if psi is not None else choice_model.psi_offer
            self._psi_value = tf.cast(psi_source, DTYPE)

        self._precompute_h = precompute_h
        self.mask_per_step = self.slate > 0                         # (T, N_t, M)

        if precompute_h:
            # Filter mode: cache the slate-only DH logits once.
            h_per_step = tf.stack(
                [tf.cast(choice_model.deephalo.compute_batch_utility(
                    None, None, tf.cast(self.slate[t], tf.float32), None), DTYPE)
                 for t in range(self.T)], axis=0,
            )                                                       # (T, N_t, M)
            self.h_per_step = h_per_step
        else:
            self.h_per_step = None

    # --- Dual-mode parameter properties -------------------------------------
    # In raw-vars mode each access recomputes the constrained value so gradient
    # flows through tanh/exp/softplus back to the raw Variables; the SSM
    # instance itself stays identical, so the @tf.function-decorated filter
    # loop is retraced only on the first call.

    @property
    def mu(self):
        if self._raw_vars_mode:
            return tf.cast(self.mu_var, DTYPE)
        return self._mu_value

    @property
    def phi(self):
        if self._raw_vars_mode:
            return tf.cast(tf.tanh(self.phi_raw_var), DTYPE)
        return self._phi_value

    @property
    def sigma(self):
        if self._raw_vars_mode:
            return tf.cast(tf.exp(self.log_sigma_var), DTYPE)
        return self._sigma_value

    @property
    def psi(self):
        if self._raw_vars_mode:
            return tf.cast(self.psi_var, DTYPE)
        return self._psi_value

    @property
    def m0(self):
        return tf.stack([self.mu / (1.0 - self.phi)])

    @property
    def P0(self):
        return tf.reshape(self.sigma ** 2 / (1.0 - self.phi ** 2), (1, 1))

    @property
    def Q(self):
        return tf.reshape(self.sigma ** 2, (1, 1))

    # --- Q2 SSM interface ---

    def f_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Deterministic AR(1) transition.

        Parameters
        ----------
        particles : tf.Tensor
            Current latent states, shape (N, 1).

        Returns
        -------
        tf.Tensor
            Propagated states ``mu + phi * particles``, shape (N, 1).
        """
        return self.mu + self.phi * particles

    def Q_sampler(self, rng: tf.random.Generator, N: int) -> tf.Tensor:
        """Sample transition noises for the AR(1) state equation.

        Parameters
        ----------
        rng : tf.random.Generator
            Random generator used to draw the standard normals.
        N : int
            Number of particles (noise samples) to draw.

        Returns
        -------
        tf.Tensor
            Scaled Gaussian noises ``sigma * eps``, shape (N, 1).
        """
        return self.sigma * rng.normal(shape=(N, 1), dtype=DTYPE)

    def log_likelihood(self, y_t: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
        """Log p(y_t | x_t = particle) summed over the N_t individuals.

        Parameters
        ----------
        y_t : tf.Tensor
            The time index ``t`` (a scalar). The Q2 PF/DPF passes ``ys[t]``,
            which we set to ``tf.range(T)``, so the actual slate/choice
            tensors are looked up by index inside this method.
        particles : tf.Tensor
            Latent states for the current step, shape (N, 1).

        Returns
        -------
        tf.Tensor
            Per-particle log-likelihood summed over the N_t individuals at
            time ``t``, shape (N,).
        """
        # particles: (N, 1) -> (N,)
        x = particles[:, 0]
        t_idx = tf.cast(tf.reshape(y_t, []), tf.int32)
        if self._precompute_h:
            h = self.h_per_step[t_idx]                              # (N_t, M)
        else:
            slate_t = tf.cast(tf.gather(self.slate, t_idx), tf.float32)
            h = tf.cast(self.choice_model.deephalo.compute_batch_utility(
                None, None, slate_t, None), DTYPE)                  # (N_t, M)
        mask = self.mask_per_step[t_idx]                            # (N_t, M)
        macro = self.psi[None, :] * x[:, None]                     # (N, M)
        utilities = h[None, :, :] + macro[:, None, :]              # (N, N_t, M)
        # Availability-masked log-softmax over the slate (no -Inf / -1e9 fill):
        # unavailable items contribute 0 to the normalizer, log-probs computed
        # directly so the DPF gradient stays finite (see common.masked_log_softmax).
        log_probs = masked_log_softmax(utilities, mask[None, :, :], axis=-1)  # (N, N_t, M)
        y = self.choice[t_idx]                                     # (N_t,)
        one_hot = tf.one_hot(y, tf.shape(utilities)[-1], dtype=DTYPE)
        log_lik_per_i = tf.reduce_sum(one_hot[None, :, :] * log_probs, axis=-1)
        return tf.reduce_sum(log_lik_per_i, axis=-1)               # (N,)

    # Helper: build the dummy ``ys`` array the Q2 filters expect.
    @property
    def ys_indices(self) -> tf.Tensor:
        """Pass this as ``ys`` to ``q2.filters.ParticleFilter.filter``."""
        return tf.cast(tf.range(self.T), DTYPE)
