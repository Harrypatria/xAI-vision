"""
Base Visualizer Module

This module provides the base class that all visualizers in the XAI-Vision
toolkit inherit from, establishing a common interface and shared functionality.

Author: Harry Patria
License: MIT
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging

# Configure logging
logger = logging.getLogger(__name__)

class BaseVisualizer(ABC):
    """
    Abstract base class for all visualizers.
    
    This class defines the common interface that all visualizers must implement
    and provides shared functionality for visualization creation and theming.
    
    Parameters
    ----------
    model_name : str, optional
        Name to use for the model in visualizations.
    height : int, default=600
        Default height for visualizations.
    width : int, default=800
        Default width for visualizations.
    
    Attributes
    ----------
    model_name : str
        Name to use for the model in visualizations.
    height : int
        Default height for visualizations.
    width : int
        Default width for visualizations.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        height: int = 600,
        width: int = 800
    ):
        """Initialize the base visualizer."""
        self.model_name = model_name or "Model"
        self.height = height
        self.width = width
    
    @abstractmethod
    def plot_feature_importance(
        self,
        features: List[str],
        importances: List[float],
        method: str = 'native',
        **kwargs
    ) -> Any:
        """
        Create a feature importance plot.
        
        Parameters
        ----------
        features : list of str
            Feature names.
        importances : list of float
            Importance values corresponding to features.
        method : str, default='native'
            Method used to compute feature importance ('native', 'permutation').
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : object
            Figure object.
        """
        pass
    
    @abstractmethod
    def plot_shap_summary(
        self,
        shap_values: Any,
        features: Any,
        feature_names: List[str],
        **kwargs
    ) -> Any:
        """
        Create a SHAP summary plot.
        
        Parameters
        ----------
        shap_values : object
            SHAP values to visualize.
        features : object
            Feature values used to compute SHAP values.
        feature_names : list of str
            Names of features.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : object
            Figure object.
        """
        pass
    
    @abstractmethod
    def plot_shap_dependence(
        self,
        shap_values: Any,
        feature_idx: int,
        feature_values: Any,
        feature_name: str,
        **kwargs
    ) -> Any:
        """
        Create a SHAP dependence plot.
        
        Parameters
        ----------
        shap_values : object
            SHAP values to visualize.
        feature_idx : int
            Index of the feature to plot.
        feature_values : object
            Values of the feature.
        feature_name : str
            Name of the feature.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : object
            Figure object.
        """
        pass
    
    @abstractmethod
    def plot_partial_dependence(
        self,
        feature_values: List[Any],
        pdp_values: List[Any],
        feature_names: List[str],
        **kwargs
    ) -> Any:
        """
        Create a partial dependence plot.
        
        Parameters
        ----------
        feature_values : list
            Grid values for each feature.
        pdp_values : list
            Partial dependence values for each feature.
        feature_names : list of str
            Names of features.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : object
            Figure object.
        """
        pass
    
    @abstractmethod
    def plot_ice_curves(
        self,
        feature_values: Any,
        ice_values: Any,
        pdp_values: Any,
        feature_name: str,
        **kwargs
    ) -> Any:
        """
        Create an Individual Conditional Expectation (ICE) plot.
        
        Parameters
        ----------
        feature_values : object
            Grid values for the feature.
        ice_values : object
            ICE values for each instance.
        pdp_values : object
            Partial dependence values (average of ICE curves).
        feature_name : str
            Name of the feature.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : object
            Figure object.
        """
        pass
    
    @abstractmethod
    def plot_decision_boundary(
        self,
        xx: Any,
        yy: Any,
        Z: Any,
        feature_x_name: str,
        feature_y_name: str,
        **kwargs
    ) -> Any:
        """
        Create a decision boundary plot.
        
        Parameters
        ----------
        xx : object
            Grid values for x-axis feature.
        yy : object
            Grid values for y-axis feature.
        Z : object
            Predicted classes or probabilities on the grid.
        feature_x_name : str
            Name of the x-axis feature.
        feature_y_name : str
            Name of the y-axis feature.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : object
            Figure object.
        """
        pass
