from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, Any, Callable, Literal, Union
import pandas as pd
import numpy as np
import optuna

from .validators import (
    validate_objective,
    validate_metric,
    validate_threshold,
    validate_input_data
)


class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that optimizes weights for an ensemble of models.
    
    This classifier takes a matrix of model probability predictions as input, where each
    column represents predictions from a different model. It then uses 
    Optuna to optimize the weights of each model in the ensemble to 
    maximize or minimize a given metric.
    
    Parameters
    ----------
    objective : str
        Either "maximize" or "minimize" the evaluation metric.
    metric : callable
        The evaluation metric function to optimize.
    n_trials : int, default=100
        Number of optimization trials.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to output optimization information.
    threshold : float, default=0.5
        Decision threshold for binary classification.
    """

    def __init__(
        self,
        objective: Literal['maximize', 'minimize'],
        metric: Callable,
        n_trials: int = 100,
        random_state: int = 42,
        verbose: bool = False,
        threshold: float = 0.5,
    ) -> None:
        self.objective = objective
        self.metric = metric
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose
        self.threshold = threshold

        validate_objective(objective)
        validate_metric(metric)
        validate_threshold(threshold)

        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'WeightedEnsembleClassifier':
        """
        Fit the weighted ensemble classifier.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix containing model probability predictions.
            Each column represents predictions from a different model.
        y : pandas.Series or numpy.ndarray
            The target values.
            
        Returns
        -------
        self : WeightedEnsembleClassifier
            Returns the instance itself.
        """
        validate_input_data(X, y)

        self.classes_ = np.unique(y)

        def objective(trial: optuna.Trial) -> float:
            weights = np.array([
                trial.suggest_float(column, 0.0, 1.0) for column in X.columns.tolist()
            ])

            weights = weights / np.sum(weights)
            y_pred_proba = np.dot(X, weights)

            if len(self.classes_) == 2:
                y_pred = (y_pred_proba >= self.threshold).astype(int)
            else:
                y_pred = y_pred_proba.argmax(axis=1)

            score = self.metric(y, y_pred)

            return score

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction=self.objective,
            sampler=sampler
        )

        study.optimize(objective, n_trials=self.n_trials)

        best_weights = np.array([
            study.best_params[column] for column in X.columns.tolist()
        ])

        self.weights_ = best_weights / np.sum(best_weights)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using the weighted ensemble classifier.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix containing model probability predictions.
            
        Returns
        -------
        numpy.ndarray
            The predicted class probabilities.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if hasattr(self, 'weights_') == False:
            raise ValueError("Estimator has not been fitted yet.")

        if len(self.classes_) == 2:
            proba = np.dot(X, self.weights_)
            return np.vstack((1 - proba, proba)).T
        else:
            return np.dot(X, self.weights_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels using the weighted ensemble classifier.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix containing model probability predictions.
            
        Returns
        -------
        numpy.ndarray
            The predicted class labels.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if hasattr(self, 'weights_') == False:
            raise ValueError("Estimator has not been fitted yet.")

        if len(self.classes_) == 2:
            proba = self.predict_proba(X)[:, 1]
            return (proba >= self.threshold).astype(int)
        else:
            return np.argmax(self.predict_proba(X), axis=1)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {
            "metric": self.metric,
            "objective": self.objective,
            "n_trials": self.n_trials,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "threshold": self.threshold,
        }

    def set_params(self, **parameters) -> 'WeightedEnsembleClassifier':
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **parameters : dict
            Estimator parameters.
            
        Returns
        -------
        self : WeightedEnsembleClassifier
            Returns the instance itself.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
