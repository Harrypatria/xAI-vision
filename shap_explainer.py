"""
SHAP Explainer Module

This module provides functionality for explaining models using SHAP
(SHapley Additive exPlanations), a game theoretic approach to explain
the output of any machine learning model.

Author: Your Name
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging
import warnings

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn(
        "SHAP package not found. SHAP explainer will not be available. "
        "Install it with: pip install shap"
    )

from .base import BaseExplainer
from ..utils.model_utils import is_tree_based, is_linear_model, is_neural_network

# Configure logging
logger = logging.getLogger(__name__)

class ShapExplainer(BaseExplainer):
    """
    Explainer for SHAP (SHapley Additive exPlanations) values.
    
    This explainer calculates SHAP values, which show the contribution of each
    feature to the prediction for each instance. It automatically selects the
    appropriate SHAP algorithm based on the model type.
    
    Parameters
    ----------
    model : object
        Trained machine learning model to explain.
    X : numpy.ndarray or pandas.DataFrame
        Training or background data used to create explanations.
    feature_names : list, optional
        Names of features in X.
    model_task : str, default='classification'
        Type of ML task: 'classification' or 'regression'.
    background_samples : int, default=100
        Number of background samples to use for explainer initialization.
    
    Attributes
    ----------
    explainer : object
        SHAP explainer object.
    shap_values : numpy.ndarray
        Computed SHAP values.
    """
    
    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        model_task: str = 'classification',
        background_samples: int = 100
    ):
        """Initialize the SHAP explainer."""
        super().__init__(model, X, feature_names)
        
        if not HAS_SHAP:
            raise ImportError(
                "SHAP package is required for this explainer. "
                "Install it with: pip install shap"
            )
        
        self.model_task = model_task
        self.background_samples = min(background_samples, len(X))
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
    
    def _select_explainer(self) -> None:
        """
        Select the appropriate SHAP explainer based on the model type.
        
        SHAP offers several explainers optimized for different model types:
        - TreeExplainer: For tree-based models (fastest)
        - LinearExplainer: For linear models
        - DeepExplainer: For neural networks
        - KernelExplainer: For any model (slowest but most general)
        """
        if is_tree_based(self.model):
            logger.info("Using TreeExplainer for tree-based model")
            self.explainer = shap.TreeExplainer(self.model)
        elif is_linear_model(self.model):
            logger.info("Using LinearExplainer for linear model")
            # For linear models, we need a subset of the data as background
            if isinstance(self.X, pd.DataFrame):
                background_data = self.X.sample(self.background_samples)
            else:
                indices = np.random.choice(
                    self.X.shape[0], 
                    size=self.background_samples, 
                    replace=False
                )
                background_data = self.X[indices]
            
            self.explainer = shap.LinearExplainer(
                self.model, 
                background_data
            )
        elif is_neural_network(self.model):
            logger.info("Using DeepExplainer for neural network")
            # For neural networks, we need a subset of the data as background
            if isinstance(self.X, pd.DataFrame):
                background_data = self.X.sample(self.background_samples).values
            else:
                indices = np.random.choice(
                    self.X.shape[0], 
                    size=self.background_samples, 
                    replace=False
                )
                background_data = self.X[indices]
            
            self.explainer = shap.DeepExplainer(
                self.model, 
                background_data
            )
        else:
            logger.info("Using KernelExplainer as fallback")
            # For any other model, use the Kernel explainer
            # This is the slowest but most general method
            if isinstance(self.X, pd.DataFrame):
                background_data = self.X.sample(self.background_samples)
            else:
                indices = np.random.choice(
                    self.X.shape[0], 
                    size=self.background_samples, 
                    replace=False
                )
                background_data = self.X[indices]
            
            # Define a prediction function
            if self.model_task == 'classification':
                # Use probability predictions for classification
                def predict_fn(X):
                    # Try different approaches to get probabilities
                    if hasattr(self.model, 'predict_proba'):
                        return self.model.predict_proba(X)
                    elif hasattr(self.model, 'predict') and hasattr(self.model, 'classes_'):
                        # Some models don't have predict_proba but still do classification
                        return np.eye(len(self.model.classes_))[self.model.predict(X).astype(int)]
                    else:
                        # Fallback: assume binary classification with 0/1 output
                        preds = self.model.predict(X)
                        return np.vstack([1 - preds, preds]).T
            else:
                # Use regular predictions for regression
                def predict_fn(X):
                    return self.model.predict(X)
            
            self.explainer = shap.KernelExplainer(
                predict_fn, 
                background_data
            )
    
    def fit(self) -> 'ShapExplainer':
        """
        Calculate SHAP values for the model.
        
        This method can be time-consuming for large datasets or complex models.
        
        Returns
        -------
        self : ShapExplainer
            Returns self.
        """
        # Select appropriate explainer
        self._select_explainer()
        
        # Compute SHAP values
        logger.info("Computing SHAP values")
        
        try:
            if isinstance(self.X, pd.DataFrame):
                self.shap_values = self.explainer.shap_values(self.X)
            else:
                self.shap_values = self.explainer.shap_values(self.X)
            
            # Store expected value
            if hasattr(self.explainer, 'expected_value'):
                self.expected_value = self.explainer.expected_value
            
            self.is_fitted = True
            logger.info("SHAP values computed successfully")
        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")
            raise
        
        return self
    
    def explain_instance(
        self,
        instance: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Explain a single instance using SHAP values.
        
        Parameters
        ----------
        instance : numpy.ndarray
            Instance to explain.
        **kwargs : dict
            Additional keyword arguments.
            
        Returns
        -------
        explanation : dict
            Dictionary containing the SHAP explanation.
        """
        if not self.is_fitted:
            self.fit()
        
        # Ensure instance is 2D
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        # Get SHAP values for this instance
        instance_shap_values = self.explainer.shap_values(instance)
        
        # For classification, SHAP may return a list of arrays (one per class)
        # We'll handle that by keeping track of all class outputs
        if isinstance(instance_shap_values, list) and len(instance_shap_values) > 1:
            # Multi-class case: create dict with class indices as keys
            shap_dict = {
                f'class_{i}': values[0] for i, values in enumerate(instance_shap_values)
            }
            
            # Add expected values if available
            if isinstance(self.expected_value, list):
                expected_dict = {
                    f'class_{i}': value for i, value in enumerate(self.expected_value)
                }
            else:
                expected_dict = {'expected_value': self.expected_value}
        else:
            # Binary classification or regression: single array
            if isinstance(instance_shap_values, list):
                # Binary classification typically returns a list with one element
                shap_dict = {'shap_values': instance_shap_values[1][0]}
                
                # Add expected value if available
                if isinstance(self.expected_value, list):
                    expected_dict = {'expected_value': self.expected_value[1]}
                else:
                    expected_dict = {'expected_value': self.expected_value}
            else:
                # Regression case
                shap_dict = {'shap_values': instance_shap_values[0]}
                expected_dict = {'expected_value': self.expected_value}
        
        # Create a dictionary with feature names and their SHAP values
        explanation = {
            'features': self.feature_names,
            'shap_values': shap_dict,
            'expected_value': expected_dict
        }
        
        return explanation
    
    def visualize_summary(
        self,
        visualizer: Any,
        **kwargs
    ) -> Any:
        """
        Create a summary plot of SHAP values.
        
        Parameters
        ----------
        visualizer : object
            Visualizer object to use for creating the plot.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        if not self.is_fitted:
            self.fit()
        
        # For classification with multiple classes, use class 1 (positive class) for binary
        # or create one plot per class for multi-class
        if isinstance(self.shap_values, list) and len(self.shap_values) > 1:
            if len(self.shap_values) == 2:  # Binary classification
                values = self.shap_values[1]  # Use positive class
                if isinstance(self.expected_value, list):
                    expected_value = self.expected_value[1]
                else:
                    expected_value = self.expected_value
            else:
                # For multi-class, just use the first class for the summary
                # (visualizer can handle multiple classes)
                values = self.shap_values
                expected_value = self.expected_value
        else:
            # Regression or binary classification with single output
            if isinstance(self.shap_values, list):
                values = self.shap_values[0]
            else:
                values = self.shap_values
            expected_value = self.expected_value
        
        # Create the visualization
        return visualizer.plot_shap_summary(
            shap_values=values,
            features=self.X,
            feature_names=self.feature_names,
            expected_value=expected_value,
            **kwargs
        )
    
    def visualize_dependence(
        self,
        visualizer: Any,
        feature_idx: int,
        feature_name: str,
        interaction_idx: Optional[int] = None,
        interaction_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Create a SHAP dependence plot for a specific feature.
        
        Parameters
        ----------
        visualizer : object
            Visualizer object to use for creating the plot.
        feature_idx : int
            Index of the feature to plot.
        feature_name : str
            Name of the feature to plot.
        interaction_idx : int, optional
            Index of the feature to use for coloring.
        interaction_name : str, optional
            Name of the feature to use for coloring.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        if not self.is_fitted:
            self.fit()
        
        # For classification with multiple classes, use class 1 (positive class) for binary
        # or create one plot per class for multi-class
        if isinstance(self.shap_values, list) and len(self.shap_values) > 1:
            if len(self.shap_values) == 2:  # Binary classification
                values = self.shap_values[1]  # Use positive class
            else:
                # For multi-class, just use the first class for the dependence plot
                # (user can specify which class to visualize)
                values = self.shap_values[0]
        else:
            # Regression or binary classification with single output
            if isinstance(self.shap_values, list):
                values = self.shap_values[0]
            else:
                values = self.shap_values
        
        # Get feature values
        if isinstance(self.X, pd.DataFrame):
            feature_values = self.X.iloc[:, feature_idx].values
            if interaction_idx is not None:
                interaction_values = self.X.iloc[:, interaction_idx].values
            else:
                interaction_values = None
        else:
            feature_values = self.X[:, feature_idx]
            if interaction_idx is not None:
                interaction_values = self.X[:, interaction_idx]
            else:
                interaction_values = None
        
        # Create the visualization
        return visualizer.plot_shap_dependence(
            shap_values=values,
            feature_idx=feature_idx,
            feature_values=feature_values,
            feature_name=feature_name,
            interaction_values=interaction_values,
            interaction_name=interaction_name,
            **kwargs
        )
