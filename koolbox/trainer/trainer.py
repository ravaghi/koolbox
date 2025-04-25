"""
Trainer module for training models with cross-validation.

This module provides the main Trainer class that extends BaseTrainer to implement
model training with cross-validation, evaluation, and prediction functionality.
"""
from sklearn.base import clone
from typing import Dict
import pandas as pd
import numpy as np
import time
import gc

from .base import BaseTrainer
from .validator import Validator


class Trainer(BaseTrainer):
    """
    Class for training models with cross-validation.
    
    This class extends BaseTrainer to implement model training with cross-validation,
    out-of-fold predictions, and ensemble predictions for test data.
    
    Inherits all parameters from BaseTrainer.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a Trainer instance.
        
        Parameters are passed to the BaseTrainer constructor.
        """
        super().__init__(*args, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series, fit_args: Dict = {}) -> None:
        """
        Fit the estimator to the data using cross-validation.
        
        Trains multiple instances of the estimator, one for each fold of the
        cross-validation split, and calculates out-of-fold predictions and scores.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series
            The target variable.
        fit_args : dict, default={}
            Additional arguments to pass to the estimator's fit method.
            
        Returns
        -------
        None
            The trained models are stored in the estimators attribute, and
            evaluation metrics are stored in fold_scores and overall_score.
        """
        Validator.validate_data_shapes(X, y)
        Validator.validate_estimator_has_method(self.estimator, "fit")

        if self.verbose:
            print(f"Training {self.estimator_name}\n")

        fold_scores = []
        oof_preds = np.zeros(X.shape[0])
        start_time = time.time()

        split = self.cv.split(X, y, **self.cv_args)
        for fold_idx, (train_idx, val_idx) in enumerate(split):
            fold_start_time = time.time()

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if self.use_early_stopping:
                fit_args["eval_set"] = [(X_val, y_val)]

            estimator = clone(self.estimator)
            estimator.fit(X_train, y_train, **fit_args)
            self.estimators.append(estimator)

            y_preds = self._get_y_preds(estimator, X_val)
            oof_preds[val_idx] = y_preds

            fold_score = self._calculate_metric(y_val, y_preds)
            fold_scores.append(fold_score)

            fold_time_taken = time.time() - fold_start_time
            if self.verbose:
                if fit_args:
                    print(
                        f"--- Fold {fold_idx} - {self.metric_name}: {fold_score:.{self.metric_precision}f} - Time: {fold_time_taken:.2f} s\n"
                    )
                else:
                    print(
                        f"--- Fold {fold_idx} - {self.metric_name}: {fold_score:.{self.metric_precision}f} - Time: {fold_time_taken:.2f} s"
                    )

            del X_train, y_train, X_val, y_val, y_preds
            gc.collect()

        overall_score = self._calculate_metric(y, oof_preds)
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        self.overall_score = overall_score
        self.fold_scores = fold_scores
        self.oof_preds = oof_preds
        self.is_fitted = True

        time_taken = time.time() - start_time
        if self.verbose:
            print(f"\n------ Overall {self.metric_name}: {overall_score:.{self.metric_precision}f} - Mean {self.metric_name}: {mean_score:.{self.metric_precision}f} ± {std_score:.{self.metric_precision}f} - Time: {time_taken:.2f} s")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data using the ensemble of trained estimators.
        
        Uses the average prediction from all estimators trained during cross-validation.
        
        Parameters
        ----------
        X_test : pandas.DataFrame
            The test feature matrix.
            
        Returns
        -------
        numpy.ndarray
            The ensemble predictions, averaged across all trained estimators.
            
        Raises
        ------
        ValueError
            If the estimator has not been fitted.
        """
        Validator.validate_is_fitted(self.is_fitted)

        test_preds = np.zeros(X_test.shape[0])
        for estimator in self.estimators:
            Validator.validate_estimator_has_method(estimator, "predict")
            test_preds += self._get_y_preds(estimator, X_test)

        return test_preds / len(self.estimators)
