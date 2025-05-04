"""
XAI-Vision: Explainable AI Visualization Toolkit
"""

from setuptools import setup, find_packages
import os
import re

# Get version from __init__.py
with open('xai_vision/__init__.py', 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        version = '0.1.0'  # Default version if not found

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Define requirements
requirements = [
    'numpy>=1.19.0',
    'pandas>=1.1.0',
    'scikit-learn>=0.24.0',
    'matplotlib>=3.3.0',
    'plotly>=4.14.0',
    'dash>=2.0.0',
    'dash-bootstrap-components>=1.0.0',
    'shap>=0.40.0',
    'lime>=0.2.0',
    'joblib>=1.0.0',
]

extras_require = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'flake8>=3.0.0',
        'black>=21.0',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'nbsphinx>=0.8.0',
    ],
    'tensorflow': ['tensorflow>=2.0.0'],
    'torch': ['torch>=1.0.0'],
    'xgboost': ['xgboost>=1.0.0'],
    'lightgbm': ['lightgbm>=3.0.0'],
    'catboost': ['catboost>=0.24.0'],
}

setup(
    name='xai-vision',
    version=version,
    author='Your Name',
    author_email='your.email@example.com',
    description='A comprehensive toolkit for generating intuitive, interactive visualizations to explain complex machine learning models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/xai-vision',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=requirements,
    extras_require=extras_require,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    keywords='explainable-ai, xai, machine-learning, visualization, interpretability, explainability',
    project_urls={
        'Documentation': 'https://xai-vision.readthedocs.io',
        'Source': 'https://github.com/yourusername/xai-vision',
        'Bug Reports': 'https://github.com/yourusername/xai-vision/issues',
    },
)
