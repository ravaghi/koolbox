"""
Validation functions for model components in the koolbox library.

This module provides validation functions to ensure correct parameter types and values
for various model components.
"""
from typing import Any, Callable
import numpy as np
import pandas as pd


def validate_objective(objective: str) -> None:
    """
    Validate the objective parameter.
    
    Parameters
    ----------
    objective : str
        The objective to validate.
        
    Raises
    ------
    ValueError
        If the objective is not 'maximize' or 'minimize'.
    """
    if objective not in ['maximize', 'minimize']:
        raise ValueError("Objective must be either 'maximize' or 'minimize'.")


def validate_direction(direction: str) -> None:
    """
    Validate the direction parameter.
    
    Parameters
    ----------
    direction : str
        The direction to validate.
        
    Raises
    ------
    ValueError
        If the direction is not 'forward' or 'backward'.
    """
    if direction not in ['forward', 'backward']:
        raise ValueError("Direction must be either 'forward' or 'backward'.")


def validate_metric(metric: Callable) -> None:
    """
    Validate the metric parameter.
    
    Parameters
    ----------
    metric : callable
        The metric function to validate.
        
    Raises
    ------
    TypeError
        If the metric is not callable.
    """
    if not callable(metric):
        raise TypeError("Metric must be a callable function.")


def validate_input_data(X: Any, y: Any) -> None:
    """
    Validate input data for model fitting.
    
    Parameters
    ----------
    X : Any
        The feature matrix.
    y : Any
        The target values.
        
    Raises
    ------
    TypeError
        If X is not a pandas DataFrame or y is not a pandas Series or numpy array.
    ValueError
        If X or y is empty.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y must be a pandas Series or numpy array.")

    if X.empty:
        raise ValueError("X cannot be empty.")

    if len(y) == 0:
        raise ValueError("y cannot be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")


def validate_threshold(threshold: float) -> None:
    """
    Validate the threshold parameter for classification.
    
    Parameters
    ----------
    threshold : float
        The threshold to validate.
        
    Raises
    ------
    ValueError
        If threshold is not between 0 and 1.
    """
    if not isinstance(threshold, (int, float)):
        raise TypeError("Threshold must be a number.")

    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1.")
