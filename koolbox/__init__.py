"""
KoolBox: A collection of helper functions designed to simplify training and inference
of machine learning models in Kaggle competitions.

This package provides utilities for:
- Easy model training with cross-validation
- Handling different types of ML tasks
- Evaluation and model validation
"""
from .trainer.trainer import Trainer
from .models.sequential_feature_selector import SequentialFeatureSelector
from .models.weighted_ensemble_regressor import WeightedEnsembleRegressor
from .models.weighted_ensemble_classifier import WeightedEnsembleClassifier

__version__ = "0.1.3"
