"""
Decision Boundary Explainer Module

This module provides functionality for visualizing decision boundaries
of machine learning models in two-dimensional feature space.

Author: Harry Patria
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging

from .base import BaseExplainer

# Configure logging
logger = logging.getLogger(__name__)

class DecisionBoundaryExplainer(BaseExplainer):
    """
    Explainer for visualizing decision boundaries.
    
    This explainer creates visualizations that show how a model divides
    the feature space into different prediction regions, providing an
    intuitive understanding of model behavior.
    
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
    class_names : list, optional
        Names of target classes for classification models.
    
    Attributes
    ----------
    class_names : list
        Names of target classes.
    """
    
    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        model_task: str = 'classification',
        class_names: Optional[List[str]] = None
    ):
        """Initialize the decision boundary explainer."""
        super().__init__(model, X, feature_names)
        self.model_task = model_task
        self.class_names = class_names
        
        # Decision boundary visualization requires at least 2 features
        if self.X.shape[1] < 2:
            raise ValueError(
                "Decision boundary visualization requires at least 2 features. "
                f"Current dataset has {self.X.shape[1]} features."
            )
    
    def fit(self) -> 'DecisionBoundaryExplainer':
        """
        Prepare the explainer.
        
        For the decision boundary explainer, this method doesn't compute
        anything in advance, as decision boundaries are computed on demand
        for specific feature pairs.
        
        Returns
        -------
        self : DecisionBoundaryExplainer
            Returns self.
        """
        self.is_fitted = True
        return self
    
    def _create_grid(
        self,
        feature_x_idx: int,
        feature_y_idx: int,
        resolution: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a grid for two features for decision boundary visualization.
        
        Parameters
        ----------
        feature_x_idx : int
            Index of the first feature (x-axis).
        feature_y_idx : int
            Index of the second feature (y-axis).
        resolution : int, default=100
            Resolution of the grid (number of points in each dimension).
            
        Returns
        -------
        xx : numpy.ndarray
            Grid values for the first feature (meshgrid).
        yy : numpy.ndarray
            Grid values for the second feature (meshgrid).
        grid_points : numpy.ndarray
            Grid points as input for model prediction.
        """
        # Extract feature columns
        if isinstance(self.X, pd.DataFrame):
            feature_x = self.X.iloc[:, feature_x_idx].values
            feature_y = self.X.iloc[:, feature_y_idx].values
        else:
            feature_x = self.X[:, feature_x_idx]
            feature_y = self.X[:, feature_y_idx]
        
        # Determine grid bounds (with a bit of padding)
        x_min, x_max = feature_x.min() - 0.1 * (feature_x.max() - feature_x.min()), feature_x.max() + 0.1 * (feature_x.max() - feature_x.min())
        y_min, y_max = feature_y.min() - 0.1 * (feature_y.max() - feature_y.min()), feature_y.max() + 0.1 * (feature_y.max() - feature_y.min())
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # Create grid points for model prediction
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        return xx, yy, grid_points
    
    def _get_model_predictions_on_grid(
        self,
        grid_points: np.ndarray,
        feature_x_idx: int,
        feature_y_idx: int
    ) -> np.ndarray:
        """
        Get model predictions on a grid of points.
        
        Parameters
        ----------
        grid_points : numpy.ndarray
            Grid points for the two features.
        feature_x_idx : int
            Index of the first feature.
        feature_y_idx : int
            Index of the second feature.
            
        Returns
        -------
        predictions : numpy.ndarray
            Model predictions on the grid.
        """
        # Create a copy of the data with mean values
        if isinstance(self.X, pd.DataFrame):
            mean_values = self.X.mean().values
            X_grid = np.tile(mean_values, (grid_points.shape[0], 1))
        else:
            mean_values = np.mean(self.X, axis=0)
            X_grid = np.tile(mean_values, (grid_points.shape[0], 1))
        
        # Replace the two features with grid values
        X_grid[:, feature_x_idx] = grid_points[:, 0]
        X_grid[:, feature_y_idx] = grid_points[:, 1]
        
        # Make predictions
        if self.model_task == 'classification':
            if hasattr(self.model, 'predict_proba'):
                # For classification with probabilities
                probabilities = self.model.predict_proba(X_grid)
                
                # For binary classification, reshape to match the grid
                if probabilities.shape[1] == 2:
                    # Return probability of positive class
                    predictions = probabilities[:, 1]
                else:
                    # For multi-class, return class labels
                    predictions = np.argmax(probabilities, axis=1)
            else:
                # For classification without probabilities
                predictions = self.model.predict(X_grid)
        else:
            # For regression
            predictions = self.model.predict(X_grid)
        
        return predictions
    
    def compute_decision_boundary(
        self,
        feature_x: Union[int, str],
        feature_y: Union[int, str],
        resolution: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute decision boundary for a pair of features.
        
        Parameters
        ----------
        feature_x : int or str
            First feature (x-axis) index or name.
        feature_y : int or str
            Second feature (y-axis) index or name.
        resolution : int, default=100
            Resolution of the grid (number of points in each dimension).
            
        Returns
        -------
        result : dict
            Dictionary containing the decision boundary information.
        """
        # Convert feature names to indices if needed
        feature_x_idx = self._get_feature_index(feature_x)
        feature_y_idx = self._get_feature_index(feature_y)
        
        # Create grid
        xx, yy, grid_points = self._create_grid(
            feature_x_idx, feature_y_idx, resolution
        )
        
        # Get model predictions on grid
        predictions = self._get_model_predictions_on_grid(
            grid_points, feature_x_idx, feature_y_idx
        )
        
        # Reshape predictions to match the grid
        if self.model_task == 'classification' and hasattr(self.model, 'predict_proba'):
            # For classification with probabilities
            try:
                # If multi-class, get probabilities for all classes
                proba = self.model.predict_proba(grid_points.reshape(-1, 2))
                Z = proba.reshape(xx.shape[0], xx.shape[1], -1)
            except Exception:
                # If binary, use the reshaped predictions
                Z = predictions.reshape(xx.shape)
        else:
            # For other cases, reshape the predictions
            Z = predictions.reshape(xx.shape)
        
        # Return decision boundary information
        result = {
            'xx': xx,
            'yy': yy,
            'Z': Z,
            'feature_x_idx': feature_x_idx,
            'feature_y_idx': feature_y_idx,
            'feature_x_name': self.feature_names[feature_x_idx],
            'feature_y_name': self.feature_names[feature_y_idx]
        }
        
        return result
    
    def visualize(
        self,
        visualizer: Any,
        feature_x: Union[int, str],
        feature_y: Union[int, str],
        feature_x_name: Optional[str] = None,
        feature_y_name: Optional[str] = None,
        resolution: int = 100,
        **kwargs
    ) -> Any:
        """
        Visualize decision boundary for a pair of features.
        
        Parameters
        ----------
        visualizer : object
            Visualizer object to use for creating the plot.
        feature_x : int or str
            First feature (x-axis) index or name.
        feature_y : int or str
            Second feature (y-axis) index or name.
        feature_x_name : str, optional
            Name to use for the first feature. If None, use the name from feature_names.
        feature_y_name : str, optional
            Name to use for the second feature. If None, use the name from feature_names.
        resolution : int, default=100
            Resolution of the grid (number of points in each dimension).
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        # Convert feature names to indices if needed
        feature_x_idx = self._get_feature_index(feature_x)
        feature_y_idx = self._get_feature_index(feature_y)
        
        # Use provided feature names or default ones
        if feature_x_name is None:
            feature_x_name = self.feature_names[feature_x_idx]
        if feature_y_name is None:
            feature_y_name = self.feature_names[feature_y_idx]
        
        # Compute decision boundary
        db_info = self.compute_decision_boundary(
            feature_x_idx, feature_y_idx, resolution
        )
        
        # Extract original data points for these features
        if isinstance(self.X, pd.DataFrame):
            X_display = np.column_stack([
                self.X.iloc[:, feature_x_idx].values,
                self.X.iloc[:, feature_y_idx].values
            ])
        else:
            X_display = np.column_stack([
                self.X[:, feature_x_idx],
                self.X[:, feature_y_idx]
            ])
        
        # Get target values if available
        y_display = kwargs.get('y', None)
        
        # Create the visualization
        fig = visualizer.plot_decision_boundary(
            xx=db_info['xx'],
            yy=db_info['yy'],
            Z=db_info['Z'],
            feature_x_name=feature_x_name,
            feature_y_name=feature_y_name,
            X=X_display,
            y=y_display,
            class_names=self.class_names,
            **kwargs
        )
        
        return fig
