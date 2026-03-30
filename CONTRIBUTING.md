# Contributing to Claude Light

Thank you for considering contributing to claude_light! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Coding Standards](#coding-standards)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Welcome contributors of all experience levels

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/claude_light.git
   cd claude_light
   ```
3. **Set up** your development environment (see below)

## How to Contribute

### Ways to Help

- 🐛 **Report bugs**: Open an issue with steps to reproduce
- 💡 **Suggest features**: Open an issue describing your idea
- 📝 **Improve documentation**: Fix typos, clarify instructions
- 🔧 **Fix bugs**: Pick an issue and submit a PR
- ✅ **Write tests**: Improve test coverage
- 🔍 **Review code**: Provide feedback on pull requests

### First Contribution

Looking for your first contribution? Check out issues labeled with:
- `good first issue` - Suitable for newcomers
- `help wanted` - Tasks the maintainers need help with

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

### Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\Activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-timeout hypothesis
```

## Testing

### Running Tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -q

# Run fuzz tests
python -m pytest tests/test_fuzz.py -v

# Run with coverage
python -m pytest --cov=claude_light tests/

# Run specific test file
python -m pytest tests/unit/test_chunking.py -v
```

### Test Requirements

- All new features should include tests
- Bug fixes should include a regression test
- Aim for meaningful coverage, not 100% for its own sake

## Pull Request Guidelines

### Before Submitting

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests** to ensure everything passes:
   ```bash
   python -m pytest -q
   ```

4. **Commit** with clear messages:
   ```bash
   git commit -m "feat: add new chunking optimization"
   ```

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated (if needed)
- [ ] No new linting errors
- [ ] Commit messages are clear
- [ ] Branch is up to date with `main`

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks
- `deps:` - Dependency updates

Example:
```
feat: add method-level chunking for Python files

- Implement tree-sitter based chunking
- Include class preamble in each chunk
- Add unit tests for chunking logic
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function signatures
- Keep functions focused (ideally < 50 lines)
- Use docstrings for public functions and classes

### Code Organization

```
claude_light/
├── claude_light.py      # Main entry point
├── claude_light/        # Package modules
│   ├── chunking.py      # Code chunking logic
│   ├── embedding.py     # Vector embeddings
│   ├── rag.py           # Retrieval logic
│   └── ...
├── tests/
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── test_fuzz.py     # Fuzz tests
└── docs/                # Documentation
```

### Documentation

- Update README.md for user-facing changes
- Add inline comments for complex logic
- Update CLAUDE.md for architectural changes

## Security Contributions

If you find a security vulnerability:
- **Do not** open a public issue
- See [SECURITY.md](SECURITY.md) for responsible disclosure

## Questions?

- Open an issue for general questions
- Check existing issues and documentation first
- Join discussions on existing issues

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (PolyForm Noncommercial 1.0.0). See [LICENSE](LICENSE) for details.
