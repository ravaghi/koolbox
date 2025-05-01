"""
Base trainer module that defines the abstract base class for all trainers.

This module provides the foundation for implementing model training with cross-validation,
metric calculation, and prediction functionality.
"""
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from typing import Callable, Dict
import numpy as np
import pandas as pd
from abc import ABC

from .validator import Validator


class BaseTrainer(ABC):
    """
    Abstract base class for trainers that implements common functionality.
    
    This class provides the foundation for training machine learning models with
    cross-validation, metric calculation, and making predictions. It handles both
    regression and classification tasks.
    
    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator to use for training.
    cv : sklearn.model_selection.BaseCrossValidator
        The cross-validation strategy to use.
    metric : callable
        The metric to use for evaluation.
    task : str
        The type of task, must be one of ['binary', 'regression'].
    metric_threshold : float, default=0.5
        The threshold to use for binary classification metrics.
    metric_precision : int, default=4
        The precision to use when printing metric values.
    metric_args : dict, default={}
        Additional arguments to pass to the metric function.
    use_early_stopping : bool, default=False
        Whether to use early stopping during training.
    cv_args : dict, default={}
        Additional arguments to pass to the cross-validator's split method.
    verbose : bool, default=True
        Whether to print progress information during training.
    """
    def __init__(
        self,
        estimator: BaseEstimator,
        cv: BaseCrossValidator,
        metric: Callable,
        task: str,
        metric_threshold: float = 0.5,
        metric_precision: int = 4,
        metric_args: Dict = {},
        use_early_stopping: bool = False,
        cv_args: Dict = {},
        verbose: bool = True
    ) -> None:
        self.estimator = estimator
        self.cv = cv
        self.metric = metric
        self.task = task
        self.metric_threshold = metric_threshold
        self.metric_precision = metric_precision
        self.metric_args = metric_args
        self.cv_args = cv_args
        self.use_early_stopping = use_early_stopping
        self.verbose = verbose

        self.estimator_name = self.estimator.__class__.__name__
        self.metric_name = self.metric.__name__
        self.overall_score = None
        self.fold_scores = None
        self.is_fitted = False
        self.estimators = []
        self.oof_preds = None

        Validator.validate_task(self.task)
        
    def _get_y_preds(self, estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from an estimator based on the task type.
        
        For regression, calls predict(). For binary classification, calls predict_proba()
        and returns the probability of the positive class.
        
        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            The estimator to use for prediction.
        X : pandas.DataFrame
            The feature matrix.
            
        Returns
        -------
        numpy.ndarray
            The predictions.
        """
        if self.task == "regression":
            return np.maximum(estimator.predict(X), 0)
        else:
            return estimator.predict_proba(X)[:, 1]
    
    def _calculate_metric(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calculate the evaluation metric.
        
        For certain metrics that require hard class labels, predictions are
        binarized using the metric_threshold.
        
        Parameters
        ----------
        y_true : pandas.Series
            The true labels.
        y_pred : numpy.ndarray
            The predicted values or probabilities.
            
        Returns
        -------
        float
            The calculated metric value.
        """
        hard_metrics = [
            "accuracy_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "fbeta_score",
            "matthews_corrcoef",
            "balanced_accuracy_score",
            "jaccard_score",
            "hamming_loss",
            "zero_one_loss",
            "cohen_kappa_score",
            "precision_recall_fscore_support",
            "fowlkes_mallows_score",
            "hinge_loss"
        ]
        
        if self.metric_name in hard_metrics:
            y_pred = (y_pred >= self.metric_threshold).astype(int)
        return self.metric(y_true, y_pred, **self.metric_args)
