# Contributing to XAI-Vision

Thank you for considering contributing to XAI-Vision! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
  - [Creating a Branch](#creating-a-branch)
  - [Making Changes](#making-changes)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Pull Requests](#pull-requests)
- [Coding Guidelines](#coding-guidelines)
  - [Code Style](#code-style)
  - [Type Hints](#type-hints)
  - [Error Handling](#error-handling)
- [Adding New Features](#adding-new-features)
  - [Explainer Modules](#explainer-modules)
  - [Visualizer Components](#visualizer-components)
- [Release Process](#release-process)
- [Communication Channels](#communication-channels)

## Code of Conduct

All contributors to XAI-Vision must adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it carefully before contributing.

## Getting Started

### Setting Up Development Environment

1. **Fork the repository**: Start by forking the XAI-Vision repository on GitHub.

2. **Clone the repository**: Clone your fork to your local machine.
   ```bash
   git clone https://github.com/yourusername/xai-vision.git
   cd xai-vision
   ```

3. **Create a virtual environment**: We recommend using a virtual environment for development.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**: Install the package in development mode with additional development dependencies.
   ```bash
   pip install -e ".[dev]"
   ```

### Project Structure

The XAI-Vision project is organized as follows:

- `xai_vision/`: Main package directory
  - `explainers/`: Explainer modules for different XAI techniques
  - `visualizers/`: Visualization components
  - `metrics/`: Interpretability metrics
  - `utils/`: Utility functions
- `examples/`: Example notebooks and scripts
- `tests/`: Test suite
- `docs/`: Documentation source files

## Development Workflow

### Creating a Branch

Create a new branch for your work. Use a descriptive branch name that reflects the changes you're making.

```bash
git checkout -b feature/your-feature-name
```

For bug fixes, use the prefix `fix/`. For enhancements, use `feature/`. For documentation, use `docs/`.

### Making Changes

1. Make your changes to the codebase.
2. Follow the coding guidelines (see below).
3. Add tests for new functionality.
4. Update the documentation if necessary.

### Testing

XAI-Vision uses pytest for testing. You can run the tests with:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=xai_vision
```

Make sure that:
- All existing tests pass
- New functionality is covered by tests
- Code coverage remains high (aim for >90%)

### Documentation

Documentation is written in reStructuredText and built using Sphinx. To build the documentation:

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`.

Documentation should include:
- Docstrings for all public functions, classes, and methods
- Examples showing how to use new features
- Theoretical background for new XAI techniques
- Updates to tutorials if relevant

### Pull Requests

1. **Push your changes**: Push your changes to your fork.
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a pull request**: Go to the [XAI-Vision repository](https://github.com/yourusername/xai-vision) and create a pull request from your branch.

3. **Describe your changes**: In the pull request, describe the changes you've made, the motivation behind them, and any relevant information for reviewers.

4. **Review process**: A maintainer will review your pull request. They may request changes or clarifications.

5. **CI checks**: Ensure that the CI checks pass (tests, linting, etc.).

6. **Merge**: Once approved, a maintainer will merge your pull request.

## Coding Guidelines

### Code Style

XAI-Vision follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. We use `black` for code formatting and `flake8` for linting.

Format your code with:
```bash
black xai_vision tests
```

And check for style issues with:
```bash
flake8 xai_vision tests
```

### Type Hints

Use type hints for function and method signatures. This improves code readability and enables better IDE support.

```python
def calculate_importance(
    model: Any,
    X: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    # Implementation
```

### Error Handling

- Use specific exception types for different error cases
- Include informative error messages
- Document exceptions in docstrings
- Catch exceptions at the appropriate level (not too high, not too low)

```python
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file.
    
    Parameters
    ----------
    file_path : str
        Path to the file.
        
    Returns
    -------
    data : pandas.DataFrame
        Loaded data.
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    # Implementation
```

## Adding New Features

### Explainer Modules

To add a new explainer module:

1. Create a new file in `xai_vision/explainers/` (e.g., `your_explainer.py`).
2. Implement a class that inherits from `BaseExplainer`.
3. Implement the required methods (`fit`, `visualize`, etc.).
4. Add the explainer to `xai_vision/explainers/__init__.py`.
5. Add tests for the new explainer in `tests/test_explainers/`.
6. Document the new explainer in the API reference.
7. Add examples demonstrating the new explainer.

### Visualizer Components

To add a new visualizer component:

1. Create a new file in `xai_vision/visualizers/` (e.g., `your_visualizer.py`).
2. Implement a class that inherits from `BaseVisualizer`.
3. Implement the required methods.
4. Add the visualizer to `xai_vision/visualizers/__init__.py`.
5. Add tests for the new visualizer in `tests/test_visualizers/`.
6. Document the new visualizer in the API reference.
7. Add examples demonstrating the new visualizer.

## Release Process

The release process for XAI-Vision is as follows:

1. **Version bump**: Update the version in `xai_vision/__init__.py`.
2. **Update changelog**: Update the changelog in `CHANGELOG.md`.
3. **Create a release branch**: Create a branch named `release/X.Y.Z`.
4. **Submit a pull request**: Submit a pull request from the release branch to `main`.
5. **Review and merge**: After review, the pull request is merged.
6. **Tag the release**: Create a tag for the release.
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
7. **Build and publish**: Build the package and publish it to PyPI.
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```

## Communication Channels

- **GitHub Issues**: Use for bug reports, feature requests, and general questions.
- **Pull Requests**: Use for code contributions and reviews.
- **Discussions**: Use for general discussions about the project, design decisions, etc.
- **Slack**: Join our Slack workspace for real-time discussions (invitation link in the GitHub repository).

Thank you for your interest in contributing to XAI-Vision!
