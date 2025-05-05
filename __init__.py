"""
XAI-Vision: Explainable AI Visualization Toolkit

A comprehensive toolkit for generating intuitive, interactive visualizations
to explain complex machine learning models.
"""

__version__ = "0.1.0"

# Import main classes
from .model_explainer import ModelExplainer

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
from .visualizers.matplotlib_viz import MatplotlibVisualizer

# Import utility functions
from .utils.model_utils import (
    check_model_compatibility,
    infer_model_task,
    is_tree_based,
    is_linear_model,
    is_neural_network
)
from .utils.data_utils import (
    check_data_compatibility,
    validate_feature_names,
    validate_class_names
)

# Define public API
__all__ = [
    "ModelExplainer",
    "FeatureImportanceExplainer",
    "ShapExplainer",
    "LimeExplainer",
    "PDPExplainer",
    "ICEExplainer",
    "DecisionBoundaryExplainer",
    "Dashboard",
    "PlotlyVisualizer",
    "MatplotlibVisualizer",
    "check_model_compatibility",
    "infer_model_task",
    "is_tree_based",
    "is_linear_model",
    "is_neural_network",
    "check_data_compatibility",
    "validate_feature_names",
    "validate_class_names"
]

# Package metadata
__author__ = "Harry Patria"
__license__ = "MIT"
__copyright__ = "Copyright 2025 Harry Patria"
