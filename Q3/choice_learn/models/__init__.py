"""Models classes and functions."""

import logging

import tensorflow as tf

from .conditional_logit import ConditionalLogit
from .deep_halo import FeaturelessDeepHalo, FeatureBasedDeepHalo
from .lu25_bayesian import Lu25Model
from .deep_halo_combined import BayesianDeepHalo
from .learning_mnl import LearningMNL
from .nested_logit import NestedLogit
from .reslogit import ResLogit
from .simple_mnl import SimpleMNL
from .tastenet import TasteNet

if len(tf.config.list_physical_devices("GPU")) > 0:
    logging.info("GPU detected, importing GPU version of RUMnet.")
    from .rumnet import GPURUMnet as RUMnet
else:
    from .rumnet import CPURUMnet as RUMnet

    logging.info("No GPU detected, importing CPU version of RUMnet.")

__all__ = [
    "ConditionalLogit",
    "FeaturelessDeepHalo",
    "FeatureBasedDeepHalo",
    "RUMnet",
    "SimpleMNL",
    "TasteNet",
    "NestedLogit",
    "ResLogit",
    "LearningMNL",
    "Lu25Model",
    "BayesianDeepHalo",
]
