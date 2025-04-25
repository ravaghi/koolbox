"""
Validator module for ensuring correct usage of the Trainer class.

This module provides validation utilities to verify inputs, estimator functionality,
and data compatibility, helping prevent common errors during the model training process.
"""
from typing import Any


class Validator:
    """
    A utility class that provides validation methods for the Trainer.
    
    This class contains class methods for validating various aspects of the training process,
    such as task type, estimator methods, model fitting status, and data shape compatibility.
    """
    SUPPORTED_TASKS = ["binary", "regression"]

    @classmethod
    def validate_task(cls, task: str) -> None:
        """
        Validate that the specified task is supported.
        
        Parameters
        ----------
        task : str
            The task type to validate, must be one of the SUPPORTED_TASKS.
            
        Raises
        ------
        ValueError
            If the task is not in the list of supported tasks.
        """
        if task not in cls.SUPPORTED_TASKS:
            raise ValueError(
                f"Task '{task}' is not supported. "
                f"Supported tasks are: {', '.join(cls.SUPPORTED_TASKS)}"
            )

    @classmethod
    def validate_is_fitted(cls, is_fitted: bool) -> None:
        """
        Validate that the estimator has been fitted.
        
        Parameters
        ----------
        is_fitted : bool
            Boolean indicating whether the estimator has been fitted.
            
        Raises
        ------
        ValueError
            If the estimator is not fitted.
        """
        if not is_fitted:
            raise ValueError(
                "Estimator is not fitted yet. Please call fit() first."
            )

    @classmethod
    def validate_estimator_has_method(cls, estimator: Any, method_name: str) -> None:
        """
        Validate that the estimator has the specified method.
        
        Parameters
        ----------
        estimator : Any
            The estimator to validate.
        method_name : str
            The name of the method to check for.
            
        Raises
        ------
        ValueError
            If the estimator does not have the specified method.
        """
        if not hasattr(estimator, method_name) or not callable(getattr(estimator, method_name)):
            raise ValueError(
                f"Estimator doesn't have the required method: {method_name}"
            )

    @classmethod
    def validate_data_shapes(cls, X, y) -> None:
        """
        Validate that the shapes of X and y are compatible.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series
            The target variable.
            
        Raises
        ------
        ValueError
            If the number of samples in X and y do not match.
        """
        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y have incompatible shapes: {X.shape} and {len(y)}"
            )
