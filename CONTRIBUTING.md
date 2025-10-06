# Contributing to Text-to-SQL Fine-tuning

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Submitting Changes](#submitting-changes)
8. [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the best solution for the project
- Help others learn and grow

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/text-to-sql-finetuning.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/text-to-sql-finetuning.git
cd text-to-sql-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install

# Install development dependencies
pip install -e ".[dev]"

# Setup environment
cp .env.example .env
# Edit .env with your credentials
```

### Pre-commit Hooks

Install pre-commit hooks for code quality:

```bash
pip install pre-commit
pre-commit install
```

## Project Structure

```
text-to-sql-finetuning/
├── config/              # Configuration management
│   └── config.py       # Config classes using dotenv
├── src/                # Core package code
│   ├── data_preparation.py  # Dataset handling
│   ├── model_setup.py       # Model initialization
│   ├── training.py          # Training logic
│   └── utils.py            # Utility functions
├── scripts/            # Executable scripts
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/              # Test files (create as needed)
└── docs/               # Additional documentation
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Use absolute imports, group by standard library, third-party, local
- **Docstrings**: Google style docstrings
- **Type hints**: Use type hints for function signatures

### Example Code Style

```python
"""
Module docstring describing the module's purpose.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)


class MyClass:
    """Class docstring describing the class."""
    
    def __init__(self, param: str):
        """
        Initialize the class.
        
        Args:
            param: Description of parameter
        """
        self.param = param
    
    def my_method(
        self,
        arg1: str,
        arg2: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Method docstring describing what the method does.
        
        Args:
            arg1: Description of arg1
            arg2: Optional description of arg2
            
        Returns:
            Dictionary with results
            
        Raises:
            ValueError: When arg1 is invalid
        """
        if not arg1:
            raise ValueError("arg1 cannot be empty")
        
        logger.info(f"Processing {arg1}")
        return {"result": arg1}
```

### Formatting Tools

Run these before committing:

```bash
# Format code with black
black src/ scripts/

# Check code style with flake8
flake8 src/ scripts/

# Type check with mypy
mypy src/ scripts/

# Sort imports
isort src/ scripts/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_preparation.py

# Run specific test
pytest tests/test_data_preparation.py::test_create_conversation
```

### Writing Tests

Create test files in `tests/` directory:

```python
import pytest
from src.data_preparation import DatasetProcessor


class TestDatasetProcessor:
    """Tests for DatasetProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a DatasetProcessor instance."""
        return DatasetProcessor("test-dataset")
    
    def test_create_conversation(self, processor):
        """Test conversation creation."""
        sample = {
            "context": "CREATE TABLE test (id INT);",
            "question": "Select all rows",
            "answer": "SELECT * FROM test;"
        }
        
        result = processor.create_conversation(sample)
        
        assert "messages" in result
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "assistant"
```

### Test Coverage Goals

- Aim for >80% code coverage
- All public functions should have tests
- Test edge cases and error conditions
- Include integration tests for key workflows

## Submitting Changes

### Pull Request Process

1. **Update your branch**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Run tests and checks**
   ```bash
   pytest
   black src/ scripts/
   flake8 src/ scripts/
   mypy src/
   ```

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

4. **Push to your fork**
   ```bash
   git push origin your-feature-branch
   ```

5. **Create Pull Request**
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

### Commit Message Format

Use conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(data): add support for custom datasets

fix(training): resolve memory leak in trainer cleanup

docs(readme): update installation instructions

test(model): add tests for model loading
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] No new warnings
```

## Areas for Contribution

### High Priority

1. **Testing**
   - Add unit tests for all modules
   - Add integration tests
   - Add performance benchmarks

2. **Documentation**
   - Improve docstrings
   - Add tutorials
   - Create video guides

3. **Features**
   - Support for more model architectures
   - Additional evaluation metrics
   - Query validation and correction
   - Multi-database support

### Medium Priority

1. **Performance**
   - Optimize data loading
   - Implement caching
   - Add batch processing

2. **Developer Experience**
   - Better error messages
   - Progress bars and logging
   - Configuration validation

3. **Deployment**
   - Docker compose setup
   - Kubernetes manifests
   - CI/CD pipelines

### Ideas Welcome

- Support for other SQL dialects
- Fine-tuning for specific domains
- Query explanation generation
- Interactive query refinement
- Web UI for testing

## Development Workflow

### Feature Development

1. Create issue describing the feature
2. Get feedback on approach
3. Implement in feature branch
4. Add tests and documentation
5. Submit pull request
6. Address review feedback
7. Merge after approval

### Bug Fix Workflow

1. Create issue describing the bug
2. Add failing test case
3. Fix the bug
4. Verify test passes
5. Submit pull request
6. Merge after approval

## Questions?

- Open an issue for questions
- Join our community discussions
- Check existing issues and PRs
- Read the documentation

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing! 🎉
