"""
Plotly Visualizer Module

This module provides functionality for creating interactive visualizations
using Plotly, a popular plotting library that generates interactive
JavaScript-based visualizations.

Author: Harry Patria
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn(
        "Plotly not found. Plotly visualizations will not be available. "
        "Install with: pip install plotly"
    )

from .base import BaseVisualizer

# Configure logging
logger = logging.getLogger(__name__)

class PlotlyVisualizer(BaseVisualizer):
    """
    Visualizer for creating interactive plots using Plotly.
    
    This class provides methods for creating various types of
    visualizations for model explanations, including feature importance,
    partial dependence plots, ICE curves, SHAP visualizations, and more.
    
    Parameters
    ----------
    model_name : str, optional
        Name to use for the model in visualizations.
    colorscale : str, default='viridis'
        Colorscale to use for visualizations.
    template : str, default='plotly_white'
        Plotly template to use for visualizations.
    height : int, default=500
        Default height for visualizations.
    width : int, default=800
        Default width for visualizations.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        colorscale: str = 'viridis',
        template: str = 'plotly_white',
        height: int = 500,
        width: int = 800
    ):
        """Initialize the Plotly visualizer."""
        if not HAS_PLOTLY:
            raise ImportError(
                "Plotly is required for this visualizer. "
                "Install with: pip install plotly"
            )
        
        self.model_name = model_name or "Model"
        self.colorscale = colorscale
        self.template = template
        self.height = height
        self.width = width
    
    def plot_feature_importance(
        self,
        features: List[str],
        importances: List[float],
        method: str = 'native',
        top_n: Optional[int] = None,
        **kwargs
    ) -> go.Figure:
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
        top_n : int, optional
            Number of top features to display.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Plotly figure object.
        """
        # Get top N features if specified
        if top_n is not None and top_n < len(features):
            # Sort by importance and get top N
            sorted_indices = np.argsort(importances)[::-1][:top_n]
            features = [features[i] for i in sorted_indices]
            importances = [importances[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add bars in ascending order of importance
        sorted_indices = np.argsort(importances)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        fig.add_trace(go.Bar(
            y=sorted_features,
            x=sorted_importances,
            orientation='h',
            marker=dict(
                color=sorted_importances,
                colorscale=self.colorscale
            )
        ))
        
        # Add title and labels
        title = f"Feature Importance ({method.capitalize()} Method)"
        method_desc = (
            "Model's built-in feature importance" if method == 'native' else
            "Permutation importance (drop in performance when feature is shuffled)"
        )
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=self.height,
            width=self.width,
            template=self.template,
            annotations=[
                dict(
                    text=method_desc,
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15,
                    xanchor='center', yanchor='top'
                )
            ]
        )
        
        return fig
    
    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        features: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
        expected_value: Optional[float] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create a SHAP summary plot.
        
        Parameters
        ----------
        shap_values : numpy.ndarray
            SHAP values to visualize.
        features : numpy.ndarray or pandas.DataFrame
            Feature values used to compute SHAP values.
        feature_names : list of str
            Names of features.
        expected_value : float, optional
            Expected value (base value) for the model.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Plotly figure object.
        """
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # For multi-class, use first class by default
            # or allow specifying class index
            class_idx = kwargs.get('class_idx', 0)
            shap_values = shap_values[class_idx]
        
        # Convert to numpy array if needed
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        # Determine feature order (by mean absolute SHAP value)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        feature_order = np.argsort(feature_importance)
        
        # Create a figure with a separate violin plot for each feature
        fig = go.Figure()
        
        # Add violin plots for each feature in order of importance
        for i in feature_order:
            # Extract feature values and corresponding SHAP values
            feature_vals = features[:, i]
            shap_vals = shap_values[:, i]
            
            # Determine feature type (categorical or continuous)
            unique_vals = np.unique(feature_vals)
            is_categorical = len(unique_vals) <= 10
            
            if is_categorical:
                # For categorical features, use box plots grouped by category
                for val in unique_vals:
                    mask = feature_vals == val
                    if np.sum(mask) > 0:  # Only add if there are values in this category
                        fig.add_trace(go.Box(
                            y=shap_vals[mask],
                            name=f"{feature_names[i]}={val}",
                            showlegend=False,
                            marker_color=px.colors.qualitative.Plotly[int(val) % len(px.colors.qualitative.Plotly)],
                            boxmean=True
                        ))
            else:
                # For continuous features, create a scatter plot
                # Color points by feature value
                fig.add_trace(go.Scatter(
                    x=feature_vals,
                    y=shap_vals,
                    mode='markers',
                    name=feature_names[i],
                    marker=dict(
                        color=feature_vals,
                        colorscale=self.colorscale,
                        showscale=True,
                        colorbar=dict(
                            title=feature_names[i]
                        )
                    ),
                    text=[f"{feature_names[i]}={val:.4g}, SHAP={shap:.4g}" 
                          for val, shap in zip(feature_vals, shap_vals)],
                    showlegend=False
                ))
        
        # Add title and labels
        title = "SHAP Summary Plot"
        if expected_value is not None:
            title += f" (Base value: {expected_value:.4g})"
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Feature Value",
            yaxis_title="SHAP Value (Impact on Model Output)",
            height=self.height,
            width=self.width,
            template=self.template
        )
        
        return fig
    
    def plot_shap_dependence(
        self,
        shap_values: np.ndarray,
        feature_idx: int,
        feature_values: np.ndarray,
        feature_name: str,
        interaction_values: Optional[np.ndarray] = None,
        interaction_name: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create a SHAP dependence plot.
        
        Parameters
        ----------
        shap_values : numpy.ndarray
            SHAP values to visualize.
        feature_idx : int
            Index of the feature to plot.
        feature_values : numpy.ndarray
            Values of the feature.
        feature_name : str
            Name of the feature.
        interaction_values : numpy.ndarray, optional
            Values of the interaction feature for coloring points.
        interaction_name : str, optional
            Name of the interaction feature.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Plotly figure object.
        """
        # Extract SHAP values for the specified feature
        shap_vals = shap_values[:, feature_idx]
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot with interaction coloring if available
        if interaction_values is not None and interaction_name is not None:
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=shap_vals,
                mode='markers',
                marker=dict(
                    color=interaction_values,
                    colorscale=self.colorscale,
                    showscale=True,
                    colorbar=dict(
                        title=interaction_name
                    )
                ),
                text=[f"{feature_name}={val:.4g}, {interaction_name}={int_val:.4g}, SHAP={shap:.4g}" 
                      for val, int_val, shap in zip(feature_values, interaction_values, shap_vals)],
                showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=shap_vals,
                mode='markers',
                marker=dict(
                    color=shap_vals,
                    colorscale=self.colorscale,
                    showscale=True,
                    colorbar=dict(
                        title="SHAP Value"
                    )
                ),
                text=[f"{feature_name}={val:.4g}, SHAP={shap:.4g}" 
                      for val, shap in zip(feature_values, shap_vals)],
                showlegend=False
            ))
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=min(feature_values),
            y0=0,
            x1=max(feature_values),
            y1=0,
            line=dict(
                color="gray",
                width=2,
                dash="dash",
            )
        )
        
        # Add title and labels
        interaction_title = f" (colored by {interaction_name})" if interaction_name else ""
        fig.update_layout(
            title=dict(
                text=f"SHAP Dependence Plot: {feature_name}{interaction_title}",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=feature_name,
            yaxis_title=f"SHAP Value (Impact on Model Output)",
            height=self.height,
            width=self.width,
            template=self.template
        )
        
        return fig
    
    def plot_partial_dependence(
        self,
        feature_values: List[np.ndarray],
        pdp_values: List[np.ndarray],
        feature_names: List[str],
        feature_types: Optional[List[str]] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create a partial dependence plot.
        
        Parameters
        ----------
        feature_values : list of numpy.ndarray
            Grid values for each feature.
        pdp_values : list of numpy.ndarray
            Partial dependence values for each feature.
        feature_names : list of str
            Names of features.
        feature_types : list of str, optional
            Types of features ('continuous' or 'categorical').
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Plotly figure object.
        """
        # Create figure with subplots
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=feature_names
        )
        
        # If feature types are not provided, infer them
        if feature_types is None:
            feature_types = []
            for values in feature_values:
                if len(np.unique(values)) <= 10:
                    feature_types.append('categorical')
                else:
                    feature_types.append('continuous')
        
        # Add traces for each feature
        for i, (values, pd_vals, name, f_type) in enumerate(zip(
            feature_values, pdp_values, feature_names, feature_types
        )):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            if f_type == 'categorical':
                fig.add_trace(
                    go.Bar(
                        x=[str(v) for v in values],
                        y=pd_vals,
                        marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                        showlegend=False
                    ),
                    row=row, col=col
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=values,
                        y=pd_vals,
                        mode='lines',
                        line=dict(
                            color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                            width=2
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            # Update axes titles
            fig.update_xaxes(title_text=name, row=row, col=col)
            
            if col == 1:
                fig.update_yaxes(title_text="Predicted Impact", row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Partial Dependence Plots",
                x=0.5,
                xanchor='center'
            ),
            height=self.height * max(1, n_rows * 0.8),
            width=self.width,
            template=self.template
        )
        
        return fig
    
    def plot_ice_curves(
        self,
        feature_values: np.ndarray,
        ice_values: np.ndarray,
        pdp_values: np.ndarray,
        feature_name: str,
        centered: bool = True,
        **kwargs
    ) -> go.Figure:
        """
        Create an Individual Conditional Expectation (ICE) plot.
        
        Parameters
        ----------
        feature_values : numpy.ndarray
            Grid values for the feature.
        ice_values : numpy.ndarray
            ICE values for each instance.
        pdp_values : numpy.ndarray
            Partial dependence values (average of ICE curves).
        feature_name : str
            Name of the feature.
        centered : bool, default=True
            Whether to center the ICE curves at the feature's minimum value.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Plotly figure object.
        """
        # Create figure
        fig = go.Figure()
        
        # Add ICE curves for each instance
        n_instances = ice_values.shape[0]
        
        # Use a subset of instances if there are too many
        max_instances = kwargs.get('max_instances', 50)
        if n_instances > max_instances:
            # Randomly sample instances
            indices = np.random.choice(n_instances, max_instances, replace=False)
        else:
            indices = range(n_instances)
        
        # Set a lower opacity for ICE curves
        ice_opacity = 0.3 if n_instances > 10 else 0.5
        
        # Add ICE curves
        for i in indices:
            values = ice_values[i, :]
            
            # Center curves if requested
            if centered:
                values = values - values[0]
            
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=values,
                mode='lines',
                line=dict(
                    color='blue',
                    width=1
                ),
                opacity=ice_opacity,
                showlegend=False
            ))
        
        # Add the average PDP curve
        pdp_values_centered = pdp_values - pdp_values[0] if centered else pdp_values
        
        fig.add_trace(go.Scatter(
            x=feature_values,
            y=pdp_values_centered,
            mode='lines',
            line=dict(
                color='red',
                width=3
            ),
            name='Average'
        ))
        
        # Add title and labels
        centered_title = " (Centered)" if centered else ""
        fig.update_layout(
            title=dict(
                text=f"ICE Curves: {feature_name}{centered_title}",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=feature_name,
            yaxis_title="Predicted Impact" + (" (Centered)" if centered else ""),
            height=self.height,
            width=self.width,
            template=self.template
        )
        
        return fig
    
    def plot_decision_boundary(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        Z: np.ndarray,
        feature_x_name: str,
        feature_y_name: str,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create a decision boundary plot.
        
        Parameters
        ----------
        xx : numpy.ndarray
            Grid values for x-axis feature.
        yy : numpy.ndarray
            Grid values for y-axis feature.
        Z : numpy.ndarray
            Predicted classes or probabilities on the grid.
        feature_x_name : str
            Name of the x-axis feature.
        feature_y_name : str
            Name of the y-axis feature.
        X : numpy.ndarray, optional
            Original data points to overlay.
        y : numpy.ndarray, optional
            Original labels for data points.
        class_names : list of str, optional
            Names of target classes.
        **kwargs : dict
            Additional keyword arguments for customization.
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Plotly figure object.
        """
        # Create figure
        fig = go.Figure()
        
        # Determine if this is a classification or regression task
        is_classification = len(Z.shape) > 2 or (len(np.unique(Z)) <= 10)
        
        if is_classification:
            # For classification, create a heatmap with discrete colors
            if len(Z.shape) > 2:
                # For multi-class with probabilities, use the predicted class
                Z_plot = np.argmax(Z, axis=2)
            else:
                # For binary or multi-class with class labels
                Z_plot = Z
            
            # Create colorscale based on number of classes
            n_classes = len(np.unique(Z_plot))
            if class_names is None:
                class_names = [f"Class {i}" for i in range(n_classes)]
            
            # Use qualitative colorscale for classification
            colorscale = []
            for i, color in enumerate(px.colors.qualitative.Plotly[:n_classes]):
                colorscale.extend([
                    [i/n_classes, color],
                    [(i+1)/n_classes, color]
                ])
            
            # Create heatmap
            fig.add_trace(go.Heatmap(
                z=Z_plot,
                x=xx[0, :],
                y=yy[:, 0],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="Class",
                    tickvals=list(range(n_classes)),
                    ticktext=class_names
                )
            ))
            
            # Add contours for decision boundaries
            fig.add_trace(go.Contour(
                z=Z_plot,
                x=xx[0, :],
                y=yy[:, 0],
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                ),
                showscale=False,
                line=dict(width=2, color='white'),
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']]
            ))
        else:
            # For regression, use a continuous colorscale
            fig.add_trace(go.Heatmap(
                z=Z,
                x=xx[0, :],
                y=yy[:, 0],
                colorscale=self.colorscale,
                showscale=True,
                colorbar=dict(
                    title="Predicted Value"
                )
            ))
        
        # Add original data points if provided
        if X is not None and y is not None:
            # For classification, color points by class
            if is_classification:
                for cls in np.unique(y):
                    mask = y == cls
                    fig.add_trace(go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        marker=dict(
                            color=px.colors.qualitative.Plotly[int(cls) % len(px.colors.qualitative.Plotly)],
                            size=8,
                            line=dict(
                                color='black',
                                width=1
                            )
                        ),
                        name=class_names[int(cls)] if class_names else f"Class {cls}"
                    ))
            else:
                # For regression, color points by actual value
                fig.add_trace(go.Scatter(
                    x=X[:, 0],
                    y=X[:, 1],
                    mode='markers',
                    marker=dict(
                        color=y,
                        colorscale=self.colorscale,
                        size=8,
                        line=dict(
                            color='black',
                            width=1
                        ),
                        showscale=True,
                        colorbar=dict(
                            title="Actual Value",
                            x=1.1
                        )
                    ),
                    showlegend=False
                ))
        
        # Add title and labels
        fig.update_layout(
            title=dict(
                text=f"Decision Boundary: {feature_x_name} vs {feature_y_name}",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=feature_x_name,
            yaxis_title=feature_y_name,
            height=self.height,
            width=self.width,
            template=self.template
        )
        
        return fig
