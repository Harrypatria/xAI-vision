# Getting Started with XAI-Vision

This guide will help you get started with XAI-Vision and quickly explore your machine learning models using powerful visualization techniques. We'll walk through a simple example from installation to creating interactive visualizations.

## Installation

XAI-Vision can be installed from PyPI:

```bash
pip install xai-vision
```

For development or to access the latest features, you can install from the source repository:

```bash
git clone https://github.com/Harrypatria/xAI-vision.git
cd xai-vision
pip install -e .
```

### Optional Dependencies

XAI-Vision works with various machine learning frameworks. You can install additional dependencies based on your needs:

```bash
# For TensorFlow/Keras support
pip install "xai-vision[tensorflow]"

# For PyTorch support
pip install "xai-vision[torch]"

# For XGBoost, LightGBM, and CatBoost support
pip install "xai-vision[xgboost,lightgbm,catboost]"

# For development (testing, documentation)
pip install "xai-vision[dev]"
```

## Quick Example

Let's start with a simple example using a RandomForest classifier on the Iris dataset:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import XAI-Vision
import xai_vision as xv

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split data and train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create an explainer
explainer = xv.ModelExplainer(
    model=model,
    X=X_train,
    y=y_train,
    feature_names=feature_names,
    class_names=class_names
)

# Fit the explainer (compute SHAP values, etc.)
explainer.fit(compute_shap_values=True)
```

## Exploring Global Model Behavior

Let's start by understanding which features are most important for the model:

```python
# Visualize feature importance
fig_importance = explainer.plot_feature_importance()
fig_importance.show()
```

Next, let's look at how each feature affects predictions with a SHAP summary plot:

```python
# Create a SHAP summary plot
fig_shap = explainer.plot_shap_summary()
fig_shap.show()
```

This visualization shows how each feature impacts the model predictions, with each point representing a single instance. Red points indicate high feature values, and blue points indicate low values. The horizontal position shows the impact on the model output.

## Understanding Feature Effects

Let's explore how specific features affect predictions using partial dependence plots:

```python
# Plot partial dependence for the most important feature
top_feature = explainer.explainers['feature_importance'].get_top_features(1)[0]
fig_pdp = explainer.plot_partial_dependence(top_feature)
fig_pdp.show()
```

We can also look at Individual Conditional Expectation (ICE) curves to see how predictions vary for individual samples:

```python
# Plot ICE curves for the same feature
fig_ice = explainer.plot_ice_curves(top_feature, num_instances=30)
fig_ice.show()
```

## Visualizing Decision Boundaries

For a more intuitive understanding of how the model divides the feature space, we can visualize the decision boundaries:

```python
# Select two important features
top_features = explainer.explainers['feature_importance'].get_top_features(2)

# Plot the decision boundary
fig_db = explainer.plot_decision_boundary(
    feature_x=top_features[0],
    feature_y=top_features[1]
)
fig_db.show()
```

## Explaining Individual Predictions

Let's explain a specific prediction:

```python
# Select an instance from the test set
sample_idx = 10
sample = X_test[sample_idx]
true_class = y_test[sample_idx]
predicted_class = model.predict([sample])[0]

print(f"True class: {class_names[true_class]}")
print(f"Predicted class: {class_names[predicted_class]}")

# Explain the prediction using LIME
fig_lime = explainer.plot_lime_explanation(sample)
fig_lime.show()
```

## Creating an Interactive Dashboard

XAI-Vision makes it easy to create an interactive dashboard for exploring your model:

```python
# Create an interactive dashboard
dashboard = explainer.explain_dashboard(
    title="Iris Classification Model Explainer",
    description="This dashboard explains the behavior of a Random Forest model for classifying Iris flowers."
)

# Launch the dashboard in a browser
dashboard.serve()
```

The dashboard provides a user-friendly interface with multiple tabs for exploring different aspects of your model:

1. **Overview**: Shows model information, feature importance, and SHAP summary
2. **Feature Analysis**: Provides partial dependence plots, ICE curves, and SHAP dependence plots for selected features
3. **Instance Explanation**: Explains individual predictions using LIME and SHAP
4. **Decision Boundaries**: Visualizes decision boundaries for selected feature pairs

## Next Steps

This getting started guide covers the basics of XAI-Vision. For more advanced usage and detailed examples, check out:

- [Example Notebooks](../examples/index.md): Detailed examples for various model types and datasets
- [API Reference](../api/index.md): Complete documentation of the XAI-Vision API
- [Explanation Methods](../theory/explanation_methods.md): In-depth explanations of the XAI techniques used in the toolkit

For specific use cases, see our domain-specific examples:
- [Healthcare Applications](../examples/use_cases/healthcare_example.md)
- [Finance Applications](../examples/use_cases/finance_example.md)
- [Manufacturing Applications](../examples/use_cases/manufacturing_example.md)
