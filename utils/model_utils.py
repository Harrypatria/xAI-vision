"""
Model Utility Functions

This module provides utility functions for handling machine learning models
from different frameworks, inferring model types, and checking compatibility.

Author: Harry Patria
License: MIT
"""

from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import numpy as np
import warnings
import logging
import inspect

# Configure logging
logger = logging.getLogger(__name__)

def check_model_compatibility(model: Any) -> bool:
    """
    Check if a model is compatible with the XAI-Vision toolkit.
    
    Parameters
    ----------
    model : object
        Machine learning model to check.
        
    Returns
    -------
    compatible : bool
        Whether the model is compatible.
        
    Raises
    ------
    ValueError
        If the model is not compatible.
    """
    # Check if model has a predict method (minimum requirement)
    if not hasattr(model, 'predict'):
        raise ValueError(
            "Model must have a predict method. "
            "Current model does not appear to be a valid ML model."
        )
    
    # Additional checks could be added for specific frameworks
    
    return True

def infer_model_task(model: Any) -> str:
    """
    Infer the type of machine learning task (classification or regression).
    
    Parameters
    ----------
    model : object
        Machine learning model to infer task from.
        
    Returns
    -------
    task_type : str
        'classification' or 'regression'.
    """
    # Check if model has predict_proba method (classification)
    if hasattr(model, 'predict_proba'):
        return 'classification'
    
    # Check if model has classes_ attribute (classification)
    if hasattr(model, 'classes_'):
        return 'classification'
    
    # Check for _estimator_type attribute (sklearn convention)
    if hasattr(model, '_estimator_type'):
        if model._estimator_type == 'classifier':
            return 'classification'
        elif model._estimator_type == 'regressor':
            return 'regression'
    
    # Check output shape from model's predict method signature
    try:
        sig = inspect.signature(model.predict)
        if 'return' in sig.return_annotation.__annotations__:
            return_type = sig.return_annotation.__annotations__['return']
            if 'float' in str(return_type).lower():
                return 'regression'
            elif 'int' in str(return_type).lower() or 'bool' in str(return_type).lower():
                return 'classification'
    except (AttributeError, ValueError, TypeError):
        pass
    
    # Default to classification
    logger.warning(
        "Could not determine model task type. "
        "Defaulting to 'classification'. "
        "Specify task_type explicitly if this is incorrect."
    )
    return 'classification'

def is_tree_based(model: Any) -> bool:
    """
    Check if a model is tree-based (e.g., Random Forest, Gradient Boosting).
    
    Parameters
    ----------
    model : object
        Machine learning model to check.
        
    Returns
    -------
    is_tree : bool
        Whether the model is tree-based.
    """
    # Check model type name
    model_type = type(model).__name__.lower()
    tree_types = [
        'randomforest', 'gradientboosting', 'extratrees', 'decisiontree',
        'xgboost', 'lightgbm', 'catboost', 'gbdt', 'gbm', 'gbregressor',
        'gbclassifier', 'xgbregressor', 'xgbclassifier'
    ]
    
    for tree_type in tree_types:
        if tree_type in model_type:
            return True
    
    # Check for specific attributes common to tree-based models
    tree_attributes = [
        'feature_importances_', 'estimators_', 'tree_', 'trees_',
        'get_booster', 'booster'
    ]
    
    for attr in tree_attributes:
        if hasattr(model, attr):
            return True
    
    # Check for XGBoost, LightGBM models specifically
    try:
        import xgboost
        if isinstance(model, xgboost.core.Booster) or isinstance(model, xgboost.sklearn.XGBModel):
            return True
    except (ImportError, AttributeError):
        pass
    
    try:
        import lightgbm
        if isinstance(model, lightgbm.basic.Booster) or isinstance(model, lightgbm.sklearn.LGBMModel):
            return True
    except (ImportError, AttributeError):
        pass
    
    try:
        import catboost
        if isinstance(model, catboost.core.CatBoost) or isinstance(model, catboost.core.CatBoostRegressor) or isinstance(model, catboost.core.CatBoostClassifier):
            return True
    except (ImportError, AttributeError):
        pass
    
    return False

def is_linear_model(model: Any) -> bool:
    """
    Check if a model is linear (e.g., Linear Regression, Logistic Regression).
    
    Parameters
    ----------
    model : object
        Machine learning model to check.
        
    Returns
    -------
    is_linear : bool
        Whether the model is linear.
    """
    # Check model type name
    model_type = type(model).__name__.lower()
    linear_types = [
        'linearregression', 'logisticregression', 'lasso', 'ridge',
        'elasticnet', 'sgd', 'linearsvc', 'linearsvr', 'linearmodel'
    ]
    
    for linear_type in linear_types:
        if linear_type in model_type:
            return True
    
    # Check for specific attributes common to linear models
    linear_attributes = ['coef_', 'intercept_']
    
    if all(hasattr(model, attr) for attr in linear_attributes):
        return True
    
    return False

def is_neural_network(model: Any) -> bool:
    """
    Check if a model is a neural network.
    
    Parameters
    ----------
    model : object
        Machine learning model to check.
        
    Returns
    -------
    is_nn : bool
        Whether the model is a neural network.
    """
    # Check for TensorFlow/Keras models
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model) or hasattr(model, 'layers'):
            return True
    except (ImportError, AttributeError):
        pass
    
    # Check for PyTorch models
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return True
    except (ImportError, AttributeError):
        pass
    
    # Check for sklearn neural network models
    model_type = type(model).__name__.lower()
    if 'mlp' in model_type or 'neural' in model_type:
        return True
    
    # Check for common neural network attributes
    nn_attributes = ['layers', 'forward', 'weights', 'parameters']
    
    for attr in nn_attributes:
        if hasattr(model, attr):
            return True
    
    return False

def get_model_feature_importances(model: Any) -> Optional[np.ndarray]:
    """
    Extract feature importances from a model if available.
    
    Parameters
    ----------
    model : object
        Machine learning model.
        
    Returns
    -------
    importances : numpy.ndarray or None
        Feature importance values if available, or None.
    """
    # Direct feature_importances_ attribute (most tree-based models)
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    
    # Linear models with coefficients
    if hasattr(model, 'coef_'):
        # For multi-class models, average across classes
        if len(model.coef_.shape) > 1:
            # Take absolute values for coefficients
            return np.mean(np.abs(model.coef_), axis=0)
        else:
            # Take absolute values for coefficients
            return np.abs(model.coef_)
    
    # XGBoost models
    try:
        import xgboost
        if isinstance(model, xgboost.core.Booster) or isinstance(model, xgboost.sklearn.XGBModel):
            if hasattr(model, 'get_score'):
                # Convert feature importance dict to array
                importances_dict = model.get_score(importance_type='gain')
                # Sort by feature index (assuming feature indices as keys)
                importances = np.zeros(len(importances_dict))
                for feat_idx, importance in importances_dict.items():
                    # Handle string feature names or integer indices
                    try:
                        idx = int(feat_idx.replace('f', ''))
                    except (ValueError, AttributeError):
                        # If feature names are strings, just use order
                        idx = list(importances_dict.keys()).index(feat_idx)
                    importances[idx] = importance
                return importances
            elif hasattr(model, 'feature_importances_'):
                return model.feature_importances_
    except (ImportError, AttributeError):
        pass
    
    # LightGBM models
    try:
        import lightgbm
        if isinstance(model, lightgbm.basic.Booster) or isinstance(model, lightgbm.sklearn.LGBMModel):
            if hasattr(model, 'feature_importance'):
                return model.feature_importance(importance_type='gain')
            elif hasattr(model, 'feature_importances_'):
                return model.feature_importances_
    except (ImportError, AttributeError):
        pass
    
    # CatBoost models
    try:
        import catboost
        if isinstance(model, catboost.core.CatBoost) or isinstance(model, catboost.core.CatBoostRegressor) or isinstance(model, catboost.core.CatBoostClassifier):
            if hasattr(model, 'get_feature_importance'):
                return model.get_feature_importance()
            elif hasattr(model, 'feature_importances_'):
                return model.feature_importances_
    except (ImportError, AttributeError):
        pass
    
    # No feature importances found
    return None

def get_model_name(model: Any) -> str:
    """
    Get a descriptive name for a model.
    
    Parameters
    ----------
    model : object
        Machine learning model.
        
    Returns
    -------
    name : str
        Descriptive name for the model.
    """
    # Start with the class name
    name = type(model).__name__
    
    # Add framework information if available
    framework = "Unknown"
    
    # Check for sklearn models
    try:
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            framework = "scikit-learn"
    except ImportError:
        pass
    
    # Check for TensorFlow/Keras models
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            framework = "TensorFlow/Keras"
    except ImportError:
        pass
    
    # Check for PyTorch models
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            framework = "PyTorch"
    except ImportError:
        pass
    
    # Check for XGBoost models
    try:
        import xgboost
        if isinstance(model, xgboost.core.Booster) or isinstance(model, xgboost.sklearn.XGBModel):
            framework = "XGBoost"
    except ImportError:
        pass
    
    # Check for LightGBM models
    try:
        import lightgbm
        if isinstance(model, lightgbm.basic.Booster) or isinstance(model, lightgbm.sklearn.LGBMModel):
            framework = "LightGBM"
    except ImportError:
        pass
    
    # Check for CatBoost models
    try:
        import catboost
        if isinstance(model, catboost.core.CatBoost) or isinstance(model, catboost.core.CatBoostRegressor) or isinstance(model, catboost.core.CatBoostClassifier):
            framework = "CatBoost"
    except ImportError:
        pass
    
    # Return formatted name
    if framework != "Unknown":
        return f"{name} ({framework})"
    else:
        return name

def get_prediction_function(model: Any, task_type: str) -> Callable:
    """
    Get a standardized prediction function for a model.
    
    Parameters
    ----------
    model : object
        Machine learning model.
    task_type : str
        'classification' or 'regression'.
        
    Returns
    -------
    predict_fn : callable
        Standardized prediction function.
    """
    if task_type == 'classification':
        # For classification, we want probability predictions
        if hasattr(model, 'predict_proba'):
            # Standard sklearn-like interface
            return model.predict_proba
        elif hasattr(model, 'predict') and hasattr(model, 'classes_'):
            # Some models don't have predict_proba but still do classification
            def predict_fn(X):
                preds = model.predict(X)
                # Convert class indices to one-hot encoding
                return np.eye(len(model.classes_))[preds.astype(int)]
            return predict_fn
        else:
            # Fallback: assume binary classification with 0/1 output
            def predict_fn(X):
                preds = model.predict(X)
                # Convert to probability-like format [P(0), P(1)]
                return np.vstack([1 - preds, preds]).T
            return predict_fn
    else:
        # For regression, just return the standard predict function
        return model.predict

def prepare_for_explanation(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Prepare feature data for explanation.
    
    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Feature data.
        
    Returns
    -------
    X_prep : numpy.ndarray
        Prepared feature data as numpy array.
    """
    # Convert to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X_prep = X.values
    else:
        X_prep = X
    
    # Handle missing values
    if np.isnan(X_prep).any():
        logger.warning("Missing values detected in feature data. Imputing with mean.")
        # Simple mean imputation for demonstration
        col_mean = np.nanmean(X_prep, axis=0)
        inds = np.where(np.isnan(X_prep))
        X_prep[inds] = np.take(col_mean, inds[1])
    
    return X_prep
