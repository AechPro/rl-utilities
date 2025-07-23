# Contributing to RL Utilities

Thank you for your interest in contributing to RL Utilities! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rl-utilities.git
   cd rl-utilities
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e .[dev]
   ```

## Making Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below.

3. **Run tests:**
   ```bash
   pytest
   ```

4. **Format code:**
   ```bash
   black rl_utilities
   ```

5. **Check code quality:**
   ```bash
   flake8 rl_utilities
   ```

## Code Style Guidelines

- Follow PEP 8 conventions
- Use type hints where appropriate
- Write clear, descriptive docstrings
- Keep functions focused and modular
- Use meaningful variable and function names

## Testing

- Write tests for new functionality
- Ensure all existing tests pass
- Include both unit tests and integration tests where appropriate
- Test with different data types (tensors, numpy arrays, lists) where applicable

## Submitting Changes

1. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** with:
   - Clear description of changes
   - Any relevant issue numbers
   - Examples of new functionality (if applicable)

## Types of Contributions

We welcome contributions including:

- **Bug fixes**
- **New logging handlers** (file logger, database logger, etc.)
- **Additional neural network modules**
- **Performance improvements**
- **Documentation improvements**
- **Example scripts and tutorials**

## Questions?

Feel free to open an issue for any questions about contributing or to discuss potential features before implementing them.

Thank you for contributing!
