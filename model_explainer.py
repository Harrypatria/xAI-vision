"""
XAI-Vision: Main model explainer class

This module contains the core ModelExplainer class that serves as the main entry point
for generating explanations and visualizations of machine learning models.

Author: Your Name
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import warnings
from pathlib import Path
import logging

# Import explainers
from .explainers.feature_importance import FeatureImportanceExplainer
from .explainers.shap_explainer import ShapExplainer
from .explainers.lime_explainer import LimeExplainer
from .explainers.pdp_explainer import PDPExplainer
from .explainers.ice_explainer import ICEExplainer
from .explainers.decision_boundary import DecisionBoundaryExplainer

# Import visualizers
from .visualizers.dashboard import Dashboard
from .visualizers.plotly_viz import PlotlyVisualizer

# Import utilities
from .utils.model_utils import (
    check_model_compatibility, 
    infer_model_task, 
    extract_feature_names
)
from .utils.data_utils import (
    check_data_compatibility, 
    validate_feature_names, 
    validate_class_names
)
from .utils.report_utils import generate_html_report

# Configure logging
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    A comprehensive class for explaining and visualizing machine learning models.
    
    The ModelExplainer serves as the main interface for the XAI-Vision toolkit,
    providing access to various explanation methods and visualization techniques.
    It supports a wide range of models from different ML frameworks and
    automatically determines the appropriate explanation methods based on the model type.
    
    Parameters
    ----------
    model : object
        Trained machine learning model to explain. Compatible with scikit-learn,
        TensorFlow/Keras, PyTorch, XGBoost, LightGBM, and CatBoost models.
    X : numpy.ndarray or pandas.DataFrame
        Training or background data used to create explanations.
    y : numpy.ndarray or pandas.Series, optional
        Target values for the training data. Required for certain explainers.
    feature_names : list, optional
        Names of features in X. If X is a DataFrame, feature names will be
        extracted automatically.
    class_names : list, optional
        Names of target classes for classification models.
    task_type : str, optional
        Type of ML task: 'classification' or 'regression'. If not specified,
        the explainer will attempt to infer it from the model.
    model_name : str, optional
        Name to use for the model in visualizations and reports.
    verbose : bool, default=False
        Whether to print detailed information during explanation generation.
    
    Attributes
    ----------
    explainers : dict
        Dictionary of initialized explainer objects.
    visualizer : object
        Main visualizer object used for creating plots.
    model_task : str
        Type of ML task: 'classification' or 'regression'.
    is_fitted : bool
        Whether the explainer has been fitted to the data.
    
    Examples
    --------
    >>> import xai_vision as xv
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> 
    >>> # Load data and train a model
    >>> data = load_iris()
    >>> X, y = data.data, data.target
    >>> feature_names = data.feature_names
    >>> class_names = data.target_names
    >>> 
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X, y)
    >>> 
    >>> # Create an explainer
    >>> explainer = xv.ModelExplainer(
    ...     model, 
    ...     X, 
    ...     y=y,
    ...     feature_names=feature_names,
    ...     class_names=class_names
    ... )
    >>> 
    >>> # Generate visualizations
    >>> explainer.plot_feature_importance()
    >>> explainer.plot_shap_summary()
    >>> 
    >>> # Create an interactive dashboard
    >>> dashboard = explainer.explain_dashboard()
    >>> dashboard.serve()
    """
    
    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        model_name: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize the ModelExplainer with a model and data."""
        self.model = model
        self.model_name = model_name if model_name else self._get_model_name()
        
        # Handle data
        self.X = X
        self.y = y
        
        # Extract feature names from DataFrame if available
        if feature_names is None and isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = feature_names
        
        self.class_names = class_names
        self.verbose = verbose
        
        # Configure logging based on verbosity
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)
        
        # Validate inputs
        self._validate_inputs()
        
        # Infer or set task type
        if task_type is None:
            self.model_task = infer_model_task(model)
            logger.info(f"Inferred task type: {self.model_task}")
        else:
            if task_type not in ['classification', 'regression']:
                raise ValueError("task_type must be 'classification' or 'regression'")
            self.model_task = task_type
        
        # Initialize explainers and visualizers
        self.explainers = {}
        self.visualizer = PlotlyVisualizer(model_name=self.model_name)
        self.is_fitted = False
        
        # Initialize explainer objects
        self._init_explainers()
        
        logger.info("ModelExplainer initialized successfully")
    
    def _validate_inputs(self):
        """Validate input data and parameters."""
        # Check model compatibility
        check_model_compatibility(self.model)
        
        # Check data compatibility
        check_data_compatibility(self.X, self.y)
        
        # Validate feature names
        if self.feature_names is not None:
            validate_feature_names(self.feature_names, self.X)
        else:
            # Generate default feature names if not provided
            n_features = self.X.shape[1]
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
            warnings.warn(
                "Feature names not provided. Using default feature names."
            )
        
        # Validate class names for classification tasks
        if self.class_names is not None:
            validate_class_names(self.class_names, self.y)
    
    def _get_model_name(self) -> str:
        """Extract model name from the model object."""
        model_type = type(self.model).__name__
        return model_type
    
    def _init_explainers(self):
        """Initialize all explainer objects."""
        # Initialize feature importance explainer
        self.explainers['feature_importance'] = FeatureImportanceExplainer(
            model=self.model,
            X=self.X,
            feature_names=self.feature_names
        )
        
        # Initialize SHAP explainer
        self.explainers['shap'] = ShapExplainer(
            model=self.model,
            X=self.X,
            feature_names=self.feature_names,
            model_task=self.model_task
        )
        
        # Initialize LIME explainer
        self.explainers['lime'] = LimeExplainer(
            model=self.model,
            X=self.X,
            feature_names=self.feature_names,
            model_task=self.model_task,
            class_names=self.class_names
        )
        
        # Initialize PDP explainer
        self.explainers['pdp'] = PDPExplainer(
            model=self.model,
            X=self.X,
            feature_names=self.feature_names,
            model_task=self.model_task
        )
        
        # Initialize ICE explainer
        self.explainers['ice'] = ICEExplainer(
            model=self.model,
            X=self.X,
            feature_names=self.feature_names,
            model_task=self.model_task
        )
        
        # Initialize decision boundary explainer for 2D visualizations
        if self.X.shape[1] >= 2:  # Only if we have at least 2 features
            self.explainers['decision_boundary'] = DecisionBoundaryExplainer(
                model=self.model,
                X=self.X,
                feature_names=self.feature_names,
                model_task=self.model_task,
                class_names=self.class_names
            )
    
    def fit(self, compute_shap_values: bool = True) -> 'ModelExplainer':
        """
        Fit all explainers to the data.
        
        This method prepares all explanation methods by computing necessary
        values such as SHAP values, which can be computationally intensive.
        
        Parameters
        ----------
        compute_shap_values : bool, default=True
            Whether to precompute SHAP values, which can be time-consuming
            for large datasets or complex models.
            
        Returns
        -------
        self : ModelExplainer
            Returns self for method chaining.
        """
        logger.info("Fitting explainers...")
        
        # Fit feature importance explainer
        self.explainers['feature_importance'].fit()
        
        # Fit SHAP explainer if requested
        if compute_shap_values:
            logger.info("Computing SHAP values (this may take some time)...")
            self.explainers['shap'].fit()
        
        # Fit other explainers
        for name, explainer in self.explainers.items():
            if name not in ['feature_importance', 'shap']:
                explainer.fit()
        
        self.is_fitted = True
        logger.info("All explainers fitted successfully")
        
        return self
    
    def plot_feature_importance(self, top_n: int = 10, **kwargs) -> Any:
        """
        Plot feature importance for the model.
        
        Parameters
        ----------
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
            self.explainers['feature_importance'].fit()
        
        return self.explainers['feature_importance'].visualize(
            visualizer=self.visualizer,
            top_n=top_n,
            **kwargs
        )
    
    def plot_shap_summary(self, **kwargs) -> Any:
        """
        Plot SHAP summary visualization.
        
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        if not self.is_fitted or not self.explainers['shap'].is_fitted:
            logger.info("Computing SHAP values (this may take some time)...")
            self.explainers['shap'].fit()
        
        return self.explainers['shap'].visualize_summary(
            visualizer=self.visualizer,
            **kwargs
        )
    
    def plot_shap_dependence(
        self, 
        feature: Union[str, int],
        interaction_feature: Optional[Union[str, int]] = None,
        **kwargs
    ) -> Any:
        """
        Plot SHAP dependence plot for a specific feature.
        
        Parameters
        ----------
        feature : str or int
            Feature to plot. Can be a feature name or index.
        interaction_feature : str or int, optional
            Feature to use for coloring points to show interaction effects.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        if not self.is_fitted or not self.explainers['shap'].is_fitted:
            logger.info("Computing SHAP values (this may take some time)...")
            self.explainers['shap'].fit()
        
        # Convert feature name to index if needed
        if isinstance(feature, str):
            feature_idx = self.feature_names.index(feature)
        else:
            feature_idx = feature
            feature = self.feature_names[feature_idx]
        
        # Convert interaction feature name to index if needed
        if interaction_feature is not None and isinstance(interaction_feature, str):
            interaction_idx = self.feature_names.index(interaction_feature)
        elif interaction_feature is not None:
            interaction_idx = interaction_feature
            interaction_feature = self.feature_names[interaction_idx]
        else:
            interaction_idx = None
        
        return self.explainers['shap'].visualize_dependence(
            visualizer=self.visualizer,
            feature_idx=feature_idx,
            feature_name=feature,
            interaction_idx=interaction_idx,
            interaction_name=interaction_feature,
            **kwargs
        )
    
    def plot_lime_explanation(
        self,
        instance: Union[np.ndarray, pd.Series, List[float]],
        num_features: int = 10,
        **kwargs
    ) -> Any:
        """
        Plot LIME explanation for a specific instance.
        
        Parameters
        ----------
        instance : array-like
            Instance to explain.
        num_features : int, default=10
            Number of features to include in the explanation.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        # Ensure instance is in the right format
        if isinstance(instance, list):
            instance = np.array(instance).reshape(1, -1)
        elif isinstance(instance, pd.Series):
            instance = instance.values.reshape(1, -1)
        elif isinstance(instance, pd.DataFrame) and len(instance) == 1:
            instance = instance.values
        elif isinstance(instance, np.ndarray) and instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        return self.explainers['lime'].explain_instance(
            instance=instance[0],  # LIME expects 1D array
            visualizer=self.visualizer,
            num_features=num_features,
            **kwargs
        )
    
    def plot_partial_dependence(
        self,
        features: Union[List[Union[str, int]], str, int],
        **kwargs
    ) -> Any:
        """
        Plot partial dependence for specified features.
        
        Parameters
        ----------
        features : list, str, or int
            Features to plot. Can be feature names or indices.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        # Convert to list if single feature
        if isinstance(features, (str, int)):
            features = [features]
        
        # Convert feature names to indices if needed
        feature_indices = []
        feature_names = []
        for feature in features:
            if isinstance(feature, str):
                feature_idx = self.feature_names.index(feature)
                feature_indices.append(feature_idx)
                feature_names.append(feature)
            else:
                feature_indices.append(feature)
                feature_names.append(self.feature_names[feature])
        
        return self.explainers['pdp'].visualize(
            visualizer=self.visualizer,
            features=feature_indices,
            feature_names=feature_names,
            **kwargs
        )
    
    def plot_ice_curves(
        self,
        feature: Union[str, int],
        num_instances: int = 50,
        centered: bool = True,
        **kwargs
    ) -> Any:
        """
        Plot Individual Conditional Expectation (ICE) curves.
        
        Parameters
        ----------
        feature : str or int
            Feature to plot. Can be a feature name or index.
        num_instances : int, default=50
            Number of instances to include in the plot.
        centered : bool, default=True
            Whether to center the ICE curves at the feature's minimum value.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        # Convert feature name to index if needed
        if isinstance(feature, str):
            feature_idx = self.feature_names.index(feature)
            feature_name = feature
        else:
            feature_idx = feature
            feature_name = self.feature_names[feature_idx]
        
        return self.explainers['ice'].visualize(
            visualizer=self.visualizer,
            feature=feature_idx,
            feature_name=feature_name,
            num_instances=num_instances,
            centered=centered,
            **kwargs
        )
    
    def plot_decision_boundary(
        self,
        feature_x: Union[str, int],
        feature_y: Union[str, int],
        resolution: int = 100,
        **kwargs
    ) -> Any:
        """
        Plot 2D decision boundary for a pair of features.
        
        Parameters
        ----------
        feature_x : str or int
            First feature for the x-axis.
        feature_y : str or int
            Second feature for the y-axis.
        resolution : int, default=100
            Resolution of the decision boundary grid.
        **kwargs : dict
            Additional keyword arguments to pass to the visualizer.
            
        Returns
        -------
        fig : object
            Plotly figure object containing the visualization.
        """
        if 'decision_boundary' not in self.explainers:
            raise ValueError(
                "Decision boundary explainer is not available. "
                "This can happen if the model does not support decision boundaries "
                "or if there are fewer than 2 features."
            )
        
        # Convert feature names to indices if needed
        if isinstance(feature_x, str):
            feature_x_idx = self.feature_names.index(feature_x)
            feature_x_name = feature_x
        else:
            feature_x_idx = feature_x
            feature_x_name = self.feature_names[feature_x_idx]
        
        if isinstance(feature_y, str):
            feature_y_idx = self.feature_names.index(feature_y)
            feature_y_name = feature_y
        else:
            feature_y_idx = feature_y
            feature_y_name = self.feature_names[feature_y_idx]
        
        return self.explainers['decision_boundary'].visualize(
            visualizer=self.visualizer,
            feature_x=feature_x_idx,
            feature_y=feature_y_idx,
            feature_x_name=feature_x_name,
            feature_y_name=feature_y_name,
            resolution=resolution,
            **kwargs
        )
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.Series, List[float]],
        explanation_types: List[str] = ['shap', 'lime'],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single instance.
        
        Parameters
        ----------
        instance : array-like
            Instance to explain.
        explanation_types : list of str, default=['shap', 'lime']
            Types of explanations to generate.
        **kwargs : dict
            Additional keyword arguments to pass to the explainers.
            
        Returns
        -------
        explanations : dict
            Dictionary containing all generated explanations.
        """
        # Ensure instance is in the right format
        if isinstance(instance, list):
            instance = np.array(instance).reshape(1, -1)
        elif isinstance(instance, pd.Series):
            instance = instance.values.reshape(1, -1)
        elif isinstance(instance, pd.DataFrame) and len(instance) == 1:
            instance = instance.values
        elif isinstance(instance, np.ndarray) and instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        explanations = {}
        
        if 'shap' in explanation_types:
            if not self.is_fitted or not self.explainers['shap'].is_fitted:
                logger.info("Computing SHAP values (this may take some time)...")
                self.explainers['shap'].fit()
            
            explanations['shap'] = self.explainers['shap'].explain_instance(
                instance=instance,
                **kwargs
            )
        
        if 'lime' in explanation_types:
            explanations['lime'] = self.explainers['lime'].explain_instance(
                instance=instance[0],  # LIME expects 1D array
                **kwargs
            )
        
        return explanations
    
    def explain_dashboard(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> Dashboard:
        """
        Create an interactive dashboard with all available explanations.
        
        Parameters
        ----------
        title : str, optional
            Title for the dashboard.
        description : str, optional
            Description text for the dashboard.
        **kwargs : dict
            Additional keyword arguments to pass to the dashboard.
            
        Returns
        -------
        dashboard : Dashboard
            Interactive dashboard object with explanation visualizations.
        """
        if title is None:
            title = f"{self.model_name} Explanation Dashboard"
        
        # Ensure all necessary explainers are fitted
        if not self.is_fitted:
            self.fit()
        
        dashboard = Dashboard(
            model=self.model,
            explainer=self,
            feature_names=self.feature_names,
            class_names=self.class_names,
            title=title,
            description=description,
            model_task=self.model_task,
            **kwargs
        )
        
        return dashboard
    
    def generate_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        include_explanations: List[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a comprehensive HTML report with model explanations.
        
        Parameters
        ----------
        output_path : str or Path, optional
            Path to save the HTML report. If not provided, the report
            will be returned as a string.
        title : str, optional
            Title for the report.
        description : str, optional
            Description text for the report.
        include_explanations : list of str, optional
            Types of explanations to include in the report.
            Default includes all available explanations.
        **kwargs : dict
            Additional keyword arguments to pass to the report generator.
            
        Returns
        -------
        report_html : str
            HTML string of the generated report. If output_path is provided,
            the report is also saved to the specified path.
        """
        if title is None:
            title = f"{self.model_name} Explanation Report"
        
        # Ensure all necessary explainers are fitted
        if not self.is_fitted:
            self.fit()
        
        # Default to all available explainers if not specified
        if include_explanations is None:
            include_explanations = list(self.explainers.keys())
        
        # Generate figures for all requested explanations
        figures = {}
        
        if 'feature_importance' in include_explanations:
            figures['feature_importance'] = self.plot_feature_importance()
        
        if 'shap' in include_explanations:
            figures['shap_summary'] = self.plot_shap_summary()
            
            # Add dependence plots for top features
            if self.explainers['shap'].is_fitted:
                top_features = self.explainers['feature_importance'].get_top_features(5)
                for feature in top_features:
                    figures[f'shap_dependence_{feature}'] = self.plot_shap_dependence(feature)
        
        if 'pdp' in include_explanations:
            top_features = self.explainers['feature_importance'].get_top_features(3)
            for feature in top_features:
                figures[f'pdp_{feature}'] = self.plot_partial_dependence(feature)
        
        if 'ice' in include_explanations:
            top_features = self.explainers['feature_importance'].get_top_features(2)
            for feature in top_features:
                figures[f'ice_{feature}'] = self.plot_ice_curves(feature)
        
        if 'decision_boundary' in include_explanations and 'decision_boundary' in self.explainers:
            # Get top 2 features for decision boundary visualization
            top_features = self.explainers['feature_importance'].get_top_features(2)
            if len(top_features) >= 2:
                figures['decision_boundary'] = self.plot_decision_boundary(
                    feature_x=top_features[0],
                    feature_y=top_features[1]
                )
        
        # Generate the HTML report
        report_html = generate_html_report(
            title=title,
            description=description,
            figures=figures,
            model_name=self.model_name,
            model_task=self.model_task,
            feature_names=self.feature_names,
            class_names=self.class_names,
            **kwargs
        )
        
        # Save to file if output path is provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            logger.info(f"Report saved to {output_path}")
        
        return report_html
    
    def __repr__(self) -> str:
        """Return string representation of the explainer."""
        return (
            f"ModelExplainer(model_name='{self.model_name}', "
            f"model_task='{self.model_task}', "
            f"num_features={len(self.feature_names)})"
        )
