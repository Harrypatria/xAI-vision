"""
Feature Importance Explainer Module

This module provides functionality for explaining models in terms of
feature importance, a fundamental concept for understanding which
features have the greatest impact on model predictions.

Author: Your Name
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging
from sklearn.inspection import permutation_importance
import warnings

from .base import BaseExplainer
from ..utils.model_utils import is_tree_based, get_model_feature_importances

# Configure logging
logger = logging.getLogger(__name__)

class FeatureImportanceExplainer(BaseExplainer):
    """
    Explainer for feature importance.
    
    This explainer calculates feature importance using model-specific methods
    when available (e.g., feature_importances_ for tree-based models) and
    falls back to permutation importance otherwise.
    
    Parameters
    ----------
    model : object
        Trained machine learning model to explain.
    X : numpy.ndarray or pandas.DataFrame
        Training or background data used to create explanations.
    y : numpy.ndarray or pandas.Series, optional
        Target values for the training data. Required for permutation importance.
    feature_names : list, optional
        Names of features in X.
    random_state : int, optional
        Random state for reproducibility in permutation importance.
    n_repeats : int, default=10
        Number of times to permute a feature when computing permutation importance.
    
    Attributes
    ----------
    importance_values : numpy.ndarray
        Computed importance values for each feature.
    importance_method : str
        Method used to compute feature importance ('native', 'permutation').
    """
    
    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        n_repeats: int = 10
    ):
        """Initialize the feature importance explainer."""
        super().__init__(model, X, feature_names)
        self.y = y
        self.random_state = random_state
        self.n_repeats = n_repeats
        self.importance_values = None
        self.importance_method = None
    
    def fit(self) -> 'FeatureImportanceExplainer':
        """
        Calculate feature importance for the model.
        
        For tree-based models, the native feature_importances_ attribute is used.
        For other models, permutation importance is calculated, which requires
        the target values y.
        
        Returns
        -------
        self : FeatureImportanceExplainer
            Returns self.
        """
        # Try to use model's native feature importance
        try:
            logger.info("Attempting to use model's native feature importance")
            self.importance_values = get_model_feature_importances(self.model)
            if self.importance_values is not None:
                self.importance_method = 'native'
                logger.info("Using model's native feature importance")
            else:
                raise AttributeError("No native feature importance found")
        except (AttributeError, TypeError):
            # Fall back to permutation importance
            logger.info("Falling back to permutation importance")
            if self.y is None:
                raise ValueError(
                    "Target values (y) are required for computing "
                    "permutation importance when the model does not "
                    "provide native feature importance."
                )
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                estimator=self.model,
                X=self.X,
                y=self.y,
                n_repeats=self.n_repeats,
                random_state=self.random_state
            )
            
            # Use mean importance
            self.importance_values = perm_importance.importances_mean
            self.importance_method = 'permutation'
            logger.info("Permutation importance calculated successfully")
        
        # Normalize importance values to sum to 1
        if self.importance_values is not None:
            # Handle zero-sum importance values
            if np.sum(np.abs(self.importance_values)) > 0:
                self.importance_values = self.importance_values / np.sum(np.abs(self.importance_values))
        
        self.is_fitted = True
        return self
    
    def get_importance_values(self) -> np.ndarray:
        """
        Get the computed feature importance values.
        
        Returns
        -------
        importance_values : numpy.ndarray
            Array of feature importance values.
        """
        if not self.is_fitted:
            self.fit()
        return self.importance_values
    
    def get_feature_ranking(self) -> List[Tuple[str, float]]:
        """
        Get features ranked by importance.
        
        Returns
        -------
        ranking : list of tuples
            List of (feature_name, importance_value) tuples sorted by importance.
        """
        if not self.is_fitted:
            self.fit()
        
        # Create a list of (feature_name, importance_value) tuples
        ranking = [(name, importance) for name, importance in
                  zip(self.feature_names, self.importance_values)]
        
        # Sort by importance in descending order
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def get_top_features(self, top_n: int = None) -> List[str]:
        """
        Get the names of the top N most important features.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top features to return. If None, all features are returned.
        
        Returns
        -------
        top_features : list of str
            Names of the top features.
        """
        ranking = self.get_feature_ranking()
        if top_n is not None:
            ranking = ranking[:top_n]
        return [name for name, _ in ranking]
    
    def visualize(
        self,
        visualizer: Any,
        top_n: int = 10,
        **kwargs
    ) -> Any:
        """
        Visualize feature importance.
        
        Parameters
        ----------
        visualizer : object
            Visualizer object to use for creating the plot.
        top_n : int, default=10
            Number of top features to display.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        if not self.is_fitted:
            self.fit()
        
        # Get feature ranking
        ranking = self.get_feature_ranking()
        
        # Limit to top N features
        if top_n is not None and top_n < len(ranking):
            ranking = ranking[:top_n]
        
        # Extract feature names and importance values
        features = [name for name, _ in ranking]
        importances = [importance for _, importance in ranking]
        
        # Create the visualization
        fig = visualizer.plot_feature_importance(
            features=features,
            importances=importances,
            method=self.importance_method,
            **kwargs
        )
        
        return fig
