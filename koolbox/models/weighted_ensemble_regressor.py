from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict, Any
import pandas as pd
import numpy as np
import optuna


class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    A regressor that optimizes weights for an ensemble of models.
    
    This regressor takes a matrix of model predictions as input, where each
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
    """

    def __init__(
        self,
        objective,
        metric,
        n_trials=100,
        random_state=42,
        verbose=False,
    ):
        self.objective = objective
        self.metric = metric
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose

        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'WeightedEnsembleRegressor':
        """
        Fit the weighted ensemble regressor.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix containing model predictions.
            Each column represents predictions from a different model.
        y : pandas.Series
            The target values.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        def objective(trial):
            weights = np.array([
                trial.suggest_float(column, 0.0, 1.0) for column in X.columns.tolist()
            ])

            weights = weights / np.sum(weights)
            y_pred = np.dot(X, weights)
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

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the weighted ensemble regressor.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix containing model predictions.
            
        Returns
        -------
        pandas.Series
            The predicted values.
        """
        return np.dot(X, self.weights_)

    def get_params(self, deep=True) -> Dict[str, Any]:
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
        }

    def set_params(self, **parameters) -> 'WeightedEnsembleRegressor':
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **parameters : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
