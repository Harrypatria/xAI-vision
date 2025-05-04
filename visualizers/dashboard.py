"""
Interactive Dashboard Module

This module provides functionality for creating interactive dashboards
that combine multiple explanation visualizations into a cohesive
interface for model exploration and interpretation.

Author: Harry Patria
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging
import uuid
import webbrowser
from pathlib import Path
import json
import tempfile
import os
import threading
import socket
import time
from datetime import datetime

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback
    import plotly.graph_objects as go
    import dash_bootstrap_components as dbc
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    logging.warning(
        "Dash packages not found. Interactive dashboards will not be available. "
        "Install with: pip install dash dash-bootstrap-components"
    )

# Configure logging
logger = logging.getLogger(__name__)

class Dashboard:
    """
    Interactive dashboard for model explanation.
    
    This class creates a Dash web application that provides an interactive
    interface for exploring various aspects of model behavior and explanations.
    
    Parameters
    ----------
    model : object
        Trained machine learning model to explain.
    explainer : object
        ModelExplainer instance with fitted explainers.
    feature_names : list
        Names of features used by the model.
    class_names : list, optional
        Names of target classes for classification models.
    title : str, default="Model Explanation Dashboard"
        Title for the dashboard.
    description : str, optional
        Description text for the dashboard.
    model_task : str, default='classification'
        Type of ML task: 'classification' or 'regression'.
    theme : str, default='BOOTSTRAP'
        Bootstrap theme to use for the dashboard.
    port : int, default=8050
        Port to use for the dashboard server.
    """
    
    def __init__(
        self,
        model: Any,
        explainer: Any,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        title: str = "Model Explanation Dashboard",
        description: Optional[str] = None,
        model_task: str = 'classification',
        theme: str = 'BOOTSTRAP',
        port: int = 8050
    ):
        """Initialize the dashboard."""
        if not HAS_DASH:
            raise ImportError(
                "Dash packages are required for interactive dashboards. "
                "Install with: pip install dash dash-bootstrap-components"
            )
        
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.class_names = class_names
        self.title = title
        self.description = description or "Explore and understand model behavior with interactive visualizations."
        self.model_task = model_task
        self.theme = getattr(dbc.themes, theme)
        self.port = port
        
        # Generate a unique ID for this dashboard instance
        self.dashboard_id = str(uuid.uuid4())[:8]
        
        # Instance to hold the Dash app
        self.app = None
        self.server_thread = None
        self.server_url = None
    
    def _find_available_port(self) -> int:
        """
        Find an available port to use for the dashboard server.
        
        Returns
        -------
        port : int
            Available port number.
        """
        # Try the specified port first
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('localhost', self.port))
            s.close()
            return self.port
        except socket.error:
            # Port is not available, find a random available port
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('localhost', 0))
            port = s.getsockname()[1]
            s.close()
            return port
    
    def _create_app(self) -> dash.Dash:
        """
        Create the Dash application for the dashboard.
        
        Returns
        -------
        app : dash.Dash
            Dash application object.
        """
        # Create the Dash app
        app = dash.Dash(
            __name__,
            external_stylesheets=[self.theme],
            suppress_callback_exceptions=True
        )
        
        # Set the app title
        app.title = self.title
        
        # Define the layout
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1(self.title, className="display-4 mb-3"),
                html.P(self.description, className="lead mb-4"),
                html.Hr()
            ], className="container mt-4"),
            
            # Main content
            html.Div([
                # Navigation tabs
                dbc.Tabs([
                    # Overview tab
                    dbc.Tab([
                        html.Div([
                            html.H3("Model Overview", className="mt-4 mb-3"),
                            html.Div([
                                html.P([
                                    html.Strong("Model Type: "), 
                                    html.Span(type(self.model).__name__)
                                ]),
                                html.P([
                                    html.Strong("Task Type: "), 
                                    html.Span(self.model_task.capitalize())
                                ]),
                                html.P([
                                    html.Strong("Number of Features: "), 
                                    html.Span(str(len(self.feature_names)))
                                ])
                            ], className="mb-4"),
                            
                            html.H4("Feature Importance", className="mt-4 mb-3"),
                            html.P(
                                "Feature importance shows which features have the "
                                "greatest impact on model predictions overall.",
                                className="mb-3"
                            ),
                            dcc.Graph(
                                id='feature-importance-plot',
                                figure=self.explainer.plot_feature_importance(top_n=15),
                                className="mb-4"
                            ),
                            
                            # Only show SHAP summary if available
                            html.Div([
                                html.H4("SHAP Summary", className="mt-4 mb-3"),
                                html.P(
                                    "SHAP values show the contribution of each feature "
                                    "to model predictions across all instances.",
                                    className="mb-3"
                                ),
                                dcc.Graph(
                                    id='shap-summary-plot',
                                    figure=self.explainer.plot_shap_summary(),
                                    className="mb-4"
                                )
                            ]) if hasattr(self.explainer.explainers.get('shap', {}), 'is_fitted') and 
                                 self.explainer.explainers['shap'].is_fitted else []
                        ], className="p-4")
                    ], label="Overview"),
                    
                    # Feature Analysis tab
                    dbc.Tab([
                        html.Div([
                            html.H3("Feature Analysis", className="mt-4 mb-3"),
                            html.P(
                                "Explore how individual features affect model predictions "
                                "using partial dependence plots and ICE curves.",
                                className="mb-3"
                            ),
                            
                            # Feature selector
                            html.Div([
                                html.Label("Select Feature:"),
                                dcc.Dropdown(
                                    id='feature-selector',
                                    options=[
                                        {'label': name, 'value': name}
                                        for name in self.feature_names
                                    ],
                                    value=self.feature_names[0],
                                    className="mb-4"
                                )
                            ]),
                            
                            # Tabs for different visualizations
                            dbc.Tabs([
                                dbc.Tab([
                                    dcc.Graph(id='pdp-plot')
                                ], label="Partial Dependence"),
                                
                                dbc.Tab([
                                    dcc.Graph(id='ice-plot')
                                ], label="ICE Curves"),
                                
                                dbc.Tab([
                                    dcc.Graph(id='shap-dependence-plot')
                                ], label="SHAP Dependence")
                            ], className="mt-3")
                        ], className="p-4")
                    ], label="Feature Analysis"),
                    
                    # Instance Explanation tab
                    dbc.Tab([
                        html.Div([
                            html.H3("Instance Explanation", className="mt-4 mb-3"),
                            html.P(
                                "Explain individual predictions by showing the "
                                "contribution of each feature to a specific prediction.",
                                className="mb-3"
                            ),
                            
                            # Instance selector
                            html.Div([
                                html.Label("Select Instance:"),
                                dcc.Dropdown(
                                    id='instance-selector',
                                    options=[
                                        {'label': f"Instance {i}", 'value': i}
                                        for i in range(min(100, len(self.explainer.X)))
                                    ],
                                    value=0,
                                    className="mb-4"
                                )
                            ]),
                            
                            # Feature values for the selected instance
                            html.Div([
                                html.H4("Feature Values", className="mt-3 mb-3"),
                                html.Div(id='instance-features')
                            ]),
                            
                            # Tabs for different explanation methods
                            dbc.Tabs([
                                dbc.Tab([
                                    dcc.Graph(id='lime-explanation-plot')
                                ], label="LIME"),
                                
                                dbc.Tab([
                                    dcc.Graph(id='shap-instance-plot')
                                ], label="SHAP")
                            ], className="mt-3")
                        ], className="p-4")
                    ], label="Instance Explanation"),
                    
                    # Decision Boundaries tab (only for models with 2D support)
                    dbc.Tab([
                        html.Div([
                            html.H3("Decision Boundaries", className="mt-4 mb-3"),
                            html.P(
                                "Visualize decision boundaries by plotting "
                                "model predictions across different feature pairs.",
                                className="mb-3"
                            ),
                            
                            # Feature pair selector
                            html.Div([
                                html.Div([
                                    html.Label("Feature X:"),
                                    dcc.Dropdown(
                                        id='feature-x-selector',
                                        options=[
                                            {'label': name, 'value': name}
                                            for name in self.feature_names
                                        ],
                                        value=self.feature_names[0],
                                        className="mb-3"
                                    )
                                ], className="col-md-6"),
                                
                                html.Div([
                                    html.Label("Feature Y:"),
                                    dcc.Dropdown(
                                        id='feature-y-selector',
                                        options=[
                                            {'label': name, 'value': name}
                                            for name in self.feature_names
                                        ],
                                        value=self.feature_names[1] if len(self.feature_names) > 1 else self.feature_names[0],
                                        className="mb-3"
                                    )
                                ], className="col-md-6")
                            ], className="row"),
                            
                            # Decision boundary plot
                            dcc.Graph(id='decision-boundary-plot')
                        ], className="p-4")
                    ], label="Decision Boundaries", disabled='decision_boundary' not in self.explainer.explainers)
                ])
            ], className="container mb-5")
        ])
        
        # Define callbacks
        
        # Feature Analysis callbacks
        @app.callback(
            Output('pdp-plot', 'figure'),
            Input('feature-selector', 'value')
        )
        def update_pdp_plot(feature):
            return self.explainer.plot_partial_dependence(feature)
        
        @app.callback(
            Output('ice-plot', 'figure'),
            Input('feature-selector', 'value')
        )
        def update_ice_plot(feature):
            return self.explainer.plot_ice_curves(feature)
        
        @app.callback(
            Output('shap-dependence-plot', 'figure'),
            Input('feature-selector', 'value')
        )
        def update_shap_dependence_plot(feature):
            if not hasattr(self.explainer.explainers.get('shap', {}), 'is_fitted') or not self.explainer.explainers['shap'].is_fitted:
                # Return empty figure if SHAP explainer is not available
                return go.Figure().update_layout(
                    title="SHAP explainer not fitted. Run explainer.fit(compute_shap_values=True) first."
                )
            return self.explainer.plot_shap_dependence(feature)
        
        # Instance Explanation callbacks
        @app.callback(
            Output('instance-features', 'children'),
            Input('instance-selector', 'value')
        )
        def update_instance_features(instance_idx):
            instance_idx = int(instance_idx)
            if isinstance(self.explainer.X, pd.DataFrame):
                instance = self.explainer.X.iloc[instance_idx]
            else:
                instance = self.explainer.X[instance_idx]
            
            # Create a table of feature values
            rows = []
            for i, (name, value) in enumerate(zip(self.feature_names, instance)):
                rows.append(
                    html.Tr([
                        html.Td(name),
                        html.Td(f"{value:.4g}" if isinstance(value, (int, float)) else str(value))
                    ])
                )
            
            return html.Div([
                html.Table(
                    [html.Thead(html.Tr([
                        html.Th("Feature"),
                        html.Th("Value")
                    ]))] +
                    [html.Tbody(rows)],
                    className="table table-striped table-sm"
                )
            ])
        
        @app.callback(
            Output('lime-explanation-plot', 'figure'),
            Input('instance-selector', 'value')
        )
        def update_lime_explanation_plot(instance_idx):
            instance_idx = int(instance_idx)
            if isinstance(self.explainer.X, pd.DataFrame):
                instance = self.explainer.X.iloc[instance_idx].values
            else:
                instance = self.explainer.X[instance_idx]
            
            return self.explainer.plot_lime_explanation(instance)
        
        @app.callback(
            Output('shap-instance-plot', 'figure'),
            Input('instance-selector', 'value')
        )
        def update_shap_instance_plot(instance_idx):
            if not hasattr(self.explainer.explainers.get('shap', {}), 'is_fitted') or not self.explainer.explainers['shap'].is_fitted:
                # Return empty figure if SHAP explainer is not available
                return go.Figure().update_layout(
                    title="SHAP explainer not fitted. Run explainer.fit(compute_shap_values=True) first."
                )
            
            instance_idx = int(instance_idx)
            if isinstance(self.explainer.X, pd.DataFrame):
                instance = self.explainer.X.iloc[instance_idx].values
            else:
                instance = self.explainer.X[instance_idx]
            
            explanation = self.explainer.explain_instance(instance, explanation_types=['shap'])
            if 'shap' not in explanation:
                return go.Figure().update_layout(
                    title="SHAP explanation not available for this instance."
                )
            
            # Create waterfall chart
            shap_values = explanation['shap']['shap_values']['shap_values'] if 'shap_values' in explanation['shap']['shap_values'] else explanation['shap']['shap_values']['class_0']
            feature_names = explanation['shap']['features']
            expected_value = explanation['shap']['expected_value']['expected_value'] if 'expected_value' in explanation['shap']['expected_value'] else explanation['shap']['expected_value']['class_0']
            
            # Sort by absolute SHAP value
            indices = np.argsort(np.abs(shap_values))[::-1]
            sorted_shap_values = [shap_values[i] for i in indices]
            sorted_feature_names = [feature_names[i] for i in indices]
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="SHAP",
                orientation="h",
                measure=["relative"] * len(sorted_shap_values) + ["total"],
                y=[f"{name} = {instance[i]:.4g}" if isinstance(instance[i], (int, float)) else f"{name} = {instance[i]}" 
                   for name, i in zip(sorted_feature_names, indices)] + ["Prediction"],
                x=sorted_shap_values + [expected_value + sum(shap_values)],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                base=expected_value
            ))
            
            fig.update_layout(
                title="SHAP Feature Contributions",
                xaxis_title="Impact on Model Output",
                yaxis_title="Features",
                height=500
            )
            
            # Add expected value as a base
            fig.add_shape(type="line",
                x0=expected_value, y0=-0.5,
                x1=expected_value, y1=len(sorted_shap_values) - 0.5,
                line=dict(color="gray", width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=expected_value,
                y=-0.5,
                text=f"E[f(X)] = {expected_value:.4g}",
                showarrow=False,
                yshift=10
            )
            
            return fig
        
        # Decision Boundaries callback
        @app.callback(
            Output('decision-boundary-plot', 'figure'),
            [Input('feature-x-selector', 'value'),
             Input('feature-y-selector', 'value')]
        )
        def update_decision_boundary_plot(feature_x, feature_y):
            if 'decision_boundary' not in self.explainer.explainers:
                return go.Figure().update_layout(
                    title="Decision boundary visualization not available for this model."
                )
            
            return self.explainer.plot_decision_boundary(
                feature_x=feature_x,
                feature_y=feature_y
            )
        
        return app
    
    def _run_server(self, debug: bool = False, use_reloader: bool = False) -> None:
        """
        Run the dashboard server in a separate thread.
        
        Parameters
        ----------
        debug : bool, default=False
            Whether to run the server in debug mode.
        use_reloader : bool, default=False
            Whether to use the reloader.
        """
        port = self._find_available_port()
        self.server_url = f"http://localhost:{port}"
        logger.info(f"Dashboard server starting at {self.server_url}")
        
        # Create and run the app
        self.app = self._create_app()
        self.app.run_server(
            debug=debug,
            port=port,
            use_reloader=use_reloader,
            host='localhost'
        )
    
    def serve(
        self,
        debug: bool = False,
        use_reloader: bool = False,
        open_browser: bool = True
    ) -> str:
        """
        Serve the dashboard.
        
        This method starts a server in a separate thread and returns
        the URL where the dashboard can be accessed.
        
        Parameters
        ----------
        debug : bool, default=False
            Whether to run the server in debug mode.
        use_reloader : bool, default=False
            Whether to use the reloader.
        open_browser : bool, default=True
            Whether to open the dashboard in a web browser.
            
        Returns
        -------
        server_url : str
            URL where the dashboard is served.
        """
        if self.server_thread is not None and self.server_thread.is_alive():
            logger.info(f"Dashboard already running at {self.server_url}")
            if open_browser:
                webbrowser.open(self.server_url)
            return self.server_url
        
        # Start server in a new thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            kwargs={'debug': debug, 'use_reloader': use_reloader}
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        logger.info(f"Dashboard running at {self.server_url}")
        
        if open_browser:
            webbrowser.open(self.server_url)
        
        return self.server_url
    
    def save_to_html(self, output_path: str) -> str:
        """
        Save the dashboard as a static HTML file.
        
        Parameters
        ----------
        output_path : str
            Path to save the HTML file.
            
        Returns
        -------
        output_path : str
            Path where the HTML file was saved.
        """
        raise NotImplementedError(
            "Saving to HTML is not yet implemented. "
            "Use serve() to run an interactive dashboard instead."
        )
