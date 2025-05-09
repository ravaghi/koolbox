"""
Sequential Feature Selector module for feature selection.

This module provides a sequential feature selection implementation that can
perform both forward and backward selection based on cross-validated model performance.
"""
from typing import Any, Callable, List, Literal, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from .validators import (
    validate_objective,
    validate_direction,
    validate_metric,
    validate_input_data
)


class SequentialFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Sequential Feature Selector that implements forward and backward selection.
    
    This class performs feature selection by iteratively selecting or removing
    features based on a specified evaluation metric and cross-validation strategy.
    
    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator to use for evaluation during feature selection.
    cv : int or cross-validation generator
        Cross-validation strategy to use for evaluating feature performance.
    objective : str
        Whether to 'maximize' or 'minimize' the metric. Use 'maximize' for metrics
        like accuracy where higher is better, and 'minimize' for metrics like error
        where lower is better.
    metric : callable
        The metric function to evaluate feature performance.
    direction : str, default='backward'
        The direction of feature selection, either 'forward' or 'backward'.
        Forward selection starts with no features and adds them one by one.
        Backward selection starts with all features and removes them one by one.
    verbose : bool, default=True
        Whether to print progress information during the selection process.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: Union[int, Any],
        objective: Literal['maximize', 'minimize'],
        metric: Callable,
        direction: Literal['forward', 'backward'] = 'backward',
        verbose: bool = True
    ) -> None:
        self.estimator = estimator
        self.cv = cv
        self.objective = objective
        self.metric = metric
        self.direction = direction
        self.verbose = verbose

        validate_objective(objective)
        validate_direction(direction)
        validate_metric(metric)

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'SequentialFeatureSelector':
        """
        Perform feature selection on the input data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or numpy.ndarray
            The target values.
            
        Returns
        -------
        self : SequentialFeatureSelector
            Returns the instance itself.
        """
        validate_input_data(X, y)

        if self.direction == 'forward':
            self.selected_features_, self.best_score_ = self._fit_forward(X, y)
        else:
            self.selected_features_, self.best_score_ = self._fit_backward(
                X, y)

        return self

    def _fit_backward(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Tuple[List[str], float]:
        """
        Perform backward feature selection.
        
        Starts with all features and iteratively removes features that improve performance
        when removed.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or numpy.ndarray
            The target values.
            
        Returns
        -------
        list
            The selected features.
        float
            The best score achieved.
        """
        all_features = X.columns.tolist()
        removed_features: List[str] = []

        best_score = self._get_score(X, y, all_features)
        if self.verbose:
            print(
                f"Baseline Score: {abs(best_score):.6f} | Feature Count: {len(all_features)}\n")

        for i, feature in enumerate(all_features):
            reduced_features = all_features.copy()
            reduced_features.remove(feature)
            reduced_features = [
                f for f in reduced_features if f not in removed_features]

            temp_score = self._get_score(X, y, reduced_features)

            if temp_score > best_score:
                best_score = temp_score
                removed_features.append(feature)
                if self.verbose:
                    print(
                        f"{i+1}/{len(all_features)} Score: {abs(temp_score):.6f} | Feature Count: {len(reduced_features)} | {feature} (New Best Score!)"
                    )
            else:
                if self.verbose:
                    print(
                        f"{i+1}/{len(all_features)} Score: {abs(temp_score):.6f} | Feature Count: {len(reduced_features)} | {feature}"
                    )

        return list(set(all_features) - set(removed_features)), best_score

    def _fit_forward(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Tuple[List[str], float]:
        """
        Perform forward feature selection.
        
        Starts with no features and iteratively adds features that improve performance
        when included.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or numpy.ndarray
            The target values.
            
        Returns
        -------
        list
            The selected features.
        float
            The best score achieved.
        """
        all_features = X.columns.tolist()
        added_features: List[str] = []

        best_score = float('-inf')

        for i, feature in enumerate(all_features):
            temp_features = added_features.copy()
            temp_features.append(feature)

            temp_score = self._get_score(X, y, temp_features)

            if temp_score > best_score:
                best_score = temp_score
                added_features.append(feature)
                if self.verbose:
                    print(
                        f"{i+1}/{len(all_features)} Score: {abs(temp_score):.6f} | Feature Count: {len(temp_features)} | {feature} (New Best Score!)"
                    )
            else:
                if self.verbose:
                    print(
                        f"{i+1}/{len(all_features)} Score: {abs(temp_score):.6f} | Feature Count: {len(temp_features)} | {feature}"
                    )

        return added_features, best_score

    def _get_score(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], feature_names: List[str]) -> float:
        """
        Calculate the cross-validated score for a subset of features.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or numpy.ndarray
            The target values.
        feature_names : list
            The names of features to include in the evaluation.
            
        Returns
        -------
        float
            The cross-validated score.
        """
        X_subset = X[feature_names].copy()

        estimator = clone(self.estimator)

        scores = cross_val_score(
            estimator,
            X_subset,
            y,
            scoring=make_scorer(self.metric),
            cv=self.cv
        )

        return -np.mean(scores) if self.objective == 'minimize' else np.mean(scores)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting only the chosen features.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
            
        Returns
        -------
        pandas.DataFrame
            Transformed data with only the selected features.
        """
        check_is_fitted(self, ['selected_features_', 'best_score_'])

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or numpy.ndarray
            The target values.
            
        Returns
        -------
        pandas.DataFrame
            Transformed data with only the selected features.
        """
        return self.fit(X, y).transform(X)

    def _more_tags(self) -> dict:
        """
        Get sklearn estimator tags.
        
        Returns
        -------
        dict
            Estimator tags indicating this transformer requires y.
        """
        return {
            'requires_y': True,
        }

    @property
    def selected_features(self) -> List[str]:
        """
        Get the selected features.
        
        Returns
        -------
        list
            The selected features.
            
        Raises
        ------
        ValueError
            If the estimator is not fitted.
        """
        check_is_fitted(self, ['selected_features_', 'best_score_'])
        return self.selected_features_

    @property
    def best_score(self) -> float:
        """
        Get the best score achieved during feature selection.
        
        Returns
        -------
        float
            The best score.
            
        Raises
        ------
        ValueError
            If the estimator is not fitted.
        """
        check_is_fitted(self, ['selected_features_', 'best_score_'])
        return self.best_score_
