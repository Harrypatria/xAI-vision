"""
Base Explainer Module

This module provides the base class that all explainers in the XAI-Vision
toolkit inherit from, establishing a common interface and shared functionality.

Author: Harry Patria
License: MIT
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging

# Configure logging
logger = logging.getLogger(__name__)

class BaseExplainer(ABC):
    """
    Abstract base class for all explainers.
    
    This class defines the common interface that all explainers must implement
    and provides shared functionality for data handling and validation.
    
    Parameters
    ----------
    model : object
        Trained machine learning model to explain.
    X : numpy.ndarray or pandas.DataFrame
        Training or background data used to create explanations.
    feature_names : list, optional
        Names of features in X.
    
    Attributes
    ----------
    is_fitted : bool
        Whether the explainer has been fitted to the data.
    """
    
    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ):
        """Initialize the base explainer."""
        self.model = model
        self.X = X
        
        # Extract feature names from DataFrame if available
        if feature_names is None and isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = feature_names
        
        # Generate default feature names if not provided
        if self.feature_names is None:
            n_features = X.shape[1]
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Ensure feature_names is a list of strings
        self.feature_names = [str(name) for name in self.feature_names]
        
        self.is_fitted = False
    
    @abstractmethod
    def fit(self) -> 'BaseExplainer':
        """
        Fit the explainer to the data.
        
        This method should be implemented by all subclasses to prepare
        the explainer for generating explanations.
        
        Returns
        -------
        self : BaseExplainer
            Returns self for method chaining.
        """
        pass
    
    def _ensure_fitted(self) -> None:
        """
        Ensure that the explainer has been fitted.
        
        Raises
        ------
        ValueError
            If the feature is not found.
        """
        if isinstance(feature, str):
            try:
                return self.feature_names.index(feature)
            except ValueError:
                raise ValueError(f"Feature '{feature}' not found in feature names")
        elif isinstance(feature, int):
            if feature < 0 or feature >= len(self.feature_names):
                raise ValueError(
                    f"Feature index {feature} out of range "
                    f"(0 to {len(self.feature_names) - 1})"
                )
            return feature
        else:
            raise TypeError(
                "Feature must be a string (feature name) or "
                "integer (feature index)"
            )
    
    def visualize(self, visualizer: Any, **kwargs) -> Any:
        """
        Generate a visualization of the explanation.
        
        This method should be implemented by subclasses to create
        visualizations specific to each explanation type.
        
        Parameters
        ----------
        visualizer : object
            Visualizer object to use for creating the plot.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Figure object containing the visualization.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the visualize method"
        )
        ------
        ValueError
            If the explainer has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} has not been fitted yet. "
                f"Call the fit() method before using this explainer."
            )
    
    def _get_feature_index(self, feature: Union[str, int]) -> int:
        """
        Get the index of a feature by name or index.
        
        Parameters
        ----------
        feature : str or int
            Feature name or index.
            
        Returns
        -------
        index : int
            Index of the feature.
            
        Raises
