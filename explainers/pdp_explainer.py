"""
Partial Dependence Plot (PDP) Explainer Module

This module provides functionality for computing and visualizing
partial dependence plots, which show the marginal effect of features
on model predictions.

Author: Harry Patria
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging
from joblib import Parallel, delayed

from .base import BaseExplainer

# Configure logging
logger = logging.getLogger(__name__)

class PDPExplainer(BaseExplainer):
    """
    Explainer for Partial Dependence Plots (PDP).
    
    PDPs show the marginal effect of a feature on the model prediction,
    averaging out the effects of all other features.
    
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
    n_grid_points : int, default=50
        Number of grid points to use for continuous features.
    percentiles : tuple, default=(0.05, 0.95)
        Lower and upper percentiles to consider for the feature range.
    n_jobs : int, default=1
        Number of jobs to run in parallel for computing PDPs.
    
    Attributes
    ----------
    feature_grids : dict
        Grid values for each feature.
    pdp_values : dict
        Computed PDP values for each feature.
    """
    
    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        model_task: str = 'classification',
        n_grid_points: int = 50,
        percentiles: Tuple[float, float] = (0.05, 0.95),
        n_jobs: int = 1
    ):
        """Initialize the PDP explainer."""
        super().__init__(model, X, feature_names)
        self.model_task = model_task
        self.n_grid_points = n_grid_points
        self.percentiles = percentiles
        self.n_jobs = n_jobs
        self.feature_grids = {}
        self.pdp_values = {}
        self.feature_types = self._infer_feature_types()
    
    def _infer_feature_types(self) -> Dict[int, str]:
        """
        Infer feature types (categorical or continuous).
        
        Returns
        -------
        feature_types : dict
            Dictionary mapping feature indices to types ('categorical' or 'continuous').
        """
        feature_types = {}
        
        for i in range(self.X.shape[1]):
            # Extract feature column
            if isinstance(self.X, pd.DataFrame):
                feature = self.X.iloc[:, i]
            else:
                feature = self.X[:, i]
            
            # Count unique values
            unique_values = np.unique(feature)
            n_unique = len(unique_values)
            
            # Determine type based on number of unique values and data type
            if n_unique <= 10 or np.issubdtype(feature.dtype, np.integer):
                feature_types[i] = 'categorical'
            else:
                feature_types[i] = 'continuous'
        
        return feature_types
    
    def _create_grid(self, feature_idx: int) -> np.ndarray:
        """
        Create a grid of values for a feature.
        
        Parameters
        ----------
        feature_idx : int
            Index of the feature.
            
        Returns
        -------
        grid : numpy.ndarray
            Grid values for the feature.
        """
        # Extract feature column
        if isinstance(self.X, pd.DataFrame):
            feature = self.X.iloc[:, feature_idx]
        else:
            feature = self.X[:, feature_idx]
        
        # For categorical features, use unique values
        if self.feature_types[feature_idx] == 'categorical':
            grid = np.sort(np.unique(feature))
        else:
            # For continuous features, create a grid within percentiles
            lower = np.percentile(feature, self.percentiles[0] * 100)
            upper = np.percentile(feature, self.percentiles[1] * 100)
            grid = np.linspace(lower, upper, self.n_grid_points)
        
        return grid
    
    def _compute_pdp_for_feature(self, feature_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute partial dependence for a single feature.
        
        Parameters
        ----------
        feature_idx : int
            Index of the feature.
            
        Returns
        -------
        grid : numpy.ndarray
            Grid values for the feature.
        pdp : numpy.ndarray
            Partial dependence values for each grid point.
        """
        # Create grid for this feature
        grid = self._create_grid(feature_idx)
        
        # Initialize array to store PDP values
        pdp = np.zeros(len(grid))
        
        # Create a copy of the data for modification
        if isinstance(self.X, pd.DataFrame):
            X_copy = self.X.copy()
        else:
            X_copy = self.X.copy()
        
        # For each grid point, compute predictions with all instances
        for i, value in enumerate(grid):
            # Replace feature value with grid point
            if isinstance(self.X, pd.DataFrame):
                X_copy.iloc[:, feature_idx] = value
                X_modified = X_copy.values
            else:
                X_modified = X_copy.copy()
                X_modified[:, feature_idx] = value
            
            # Make predictions
            if self.model_task == 'classification':
                # For classification, use probability of positive class
                if hasattr(self.model, 'predict_proba'):
                    predictions = self.model.predict_proba(X_modified)
                    # Take probability of positive class (for binary) or mean of all classes
                    if predictions.shape[1] == 2:
                        pdp[i] = np.mean(predictions[:, 1])
                    else:
                        pdp[i] = np.mean(predictions)
                else:
                    # Fall back to regular predict
                    predictions = self.model.predict(X_modified)
                    pdp[i] = np.mean(predictions)
            else:
                # For regression, use regular predictions
                predictions = self.model.predict(X_modified)
                pdp[i] = np.mean(predictions)
        
        return grid, pdp
    
    def _compute_pdp_parallel(self, feature_idx: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Compute PDP for a feature, to be used with parallel processing.
        
        Parameters
        ----------
        feature_idx : int
            Index of the feature.
            
        Returns
        -------
        feature_idx : int
            Index of the feature.
        grid : numpy.ndarray
            Grid values for the feature.
        pdp : numpy.ndarray
            Partial dependence values for each grid point.
        """
        grid, pdp = self._compute_pdp_for_feature(feature_idx)
        return feature_idx, grid, pdp
    
    def fit(
        self, 
        features: Optional[List[int]] = None
    ) -> 'PDPExplainer':
        """
        Compute partial dependence plots for features.
        
        Parameters
        ----------
        features : list of int, optional
            Indices of features to compute PDPs for.
            If None, compute for all features.
            
        Returns
        -------
        self : PDPExplainer
            Returns self.
        """
        # Determine features to compute PDPs for
        if features is None:
            features = list(range(self.X.shape[1]))
        
        # Log the computation
        logger.info(f"Computing PDPs for {len(features)} features")
        
        # Use parallel processing if requested
        if self.n_jobs != 1 and len(features) > 1:
            # Compute PDPs in parallel
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_pdp_parallel)(feature_idx)
                for feature_idx in features
            )
            
            # Process results
            for feature_idx, grid, pdp in results:
                self.feature_grids[feature_idx] = grid
                self.pdp_values[feature_idx] = pdp
        else:
            # Compute PDPs sequentially
            for feature_idx in features:
                grid, pdp = self._compute_pdp_for_feature(feature_idx)
                self.feature_grids[feature_idx] = grid
                self.pdp_values[feature_idx] = pdp
        
        self.is_fitted = True
        logger.info("PDP computation completed")
        
        return self
    
    def get_pdp(
        self, 
        feature: Union[int, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the partial dependence for a feature.
        
        Parameters
        ----------
        feature : int or str
            Feature index or name.
            
        Returns
        -------
        grid : numpy.ndarray
            Grid values for the feature.
        pdp : numpy.ndarray
            Partial dependence values for each grid point.
        """
        # Convert feature name to index if needed
        feature_idx = self._get_feature_index(feature)
        
        # Compute PDP if not already computed
        if feature_idx not in self.feature_grids or feature_idx not in self.pdp_values:
            grid, pdp = self._compute_pdp_for_feature(feature_idx)
            self.feature_grids[feature_idx] = grid
            self.pdp_values[feature_idx] = pdp
        
        return self.feature_grids[feature_idx], self.pdp_values[feature_idx]
    
    def visualize(
        self,
        visualizer: Any,
        features: Union[List[Union[int, str]], int, str],
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Visualize partial dependence plots.
        
        Parameters
        ----------
        visualizer : object
            Visualizer object to use for creating the plot.
        features : list, int, or str
            Features to plot. Can be feature indices or names.
        feature_names : list of str, optional
            Names to use for the features. If None, use self.feature_names.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        # Ensure features is a list
        if not isinstance(features, list):
            features = [features]
        
        # Convert feature names to indices if needed
        feature_indices = []
        for feature in features:
            feature_indices.append(self._get_feature_index(feature))
        
        # Use provided feature names or default ones
        if feature_names is None:
            feature_names = [self.feature_names[idx] for idx in feature_indices]
        
        # Make sure all features have PDP values
        for feature_idx in feature_indices:
            if feature_idx not in self.feature_grids or feature_idx not in self.pdp_values:
                self.get_pdp(feature_idx)
        
        # Prepare data for visualization
        feature_values = [self.feature_grids[idx] for idx in feature_indices]
        pdp_values = [self.pdp_values[idx] for idx in feature_indices]
        feature_types = [self.feature_types[idx] for idx in feature_indices]
        
        # Create visualization
        return visualizer.plot_partial_dependence(
            feature_values=feature_values,
            pdp_values=pdp_values,
            feature_names=feature_names,
            feature_types=feature_types,
            **kwargs
        )
