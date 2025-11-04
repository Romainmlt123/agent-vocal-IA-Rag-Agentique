# Contributing to Agent Vocal Prof

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/intelligence_lab_agent_vocal.git
   cd intelligence_lab_agent_vocal
   ```
3. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following the code style guidelines
3. **Run tests**:
   ```bash
   pytest tests/
   ```
4. **Run linter**:
   ```bash
   flake8 src/ tests/
   black src/ tests/ --check
   ```
5. **Commit your changes**:
   ```bash
   git commit -m "feat: add your feature description"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request** on GitHub

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular
- Add unit tests for new functionality

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Commit Messages

Follow conventional commit format:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code refactoring
- `chore:` maintenance tasks

## Questions?

Open an issue on GitHub for any questions or discussions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
