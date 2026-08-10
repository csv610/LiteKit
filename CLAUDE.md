# CLAUDE.md - AI Assistant Guide for LiteLLM

**Last Updated**: 2025-11-30
**Version**: 0.1.0
**Python Version**: 3.8+ (3.12 preferred)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Design Patterns](#architecture--design-patterns)
3. [Codebase Structure](#codebase-structure)
4. [Development Workflows](#development-workflows)
5. [Coding Conventions](#coding-conventions)
6. [Testing Guidelines](#testing-guidelines)
7. [Common Tasks](#common-tasks)
8. [Git Workflow](#git-workflow)
9. [Key Patterns to Follow](#key-patterns-to-follow)
10. [Environment & Dependencies](#environment--dependencies)

---

## Project Overview

### Purpose
LiteLLM is a unified CLI and library for interacting with multiple Large Language Model providers (OpenAI, Ollama, Google Gemini, Perplexity) through a single, consistent interface. It supports both text generation and vision/image analysis capabilities.

### Key Features
- **Unified Interface**: Single `LiteClient` class handles both text and vision operations
- **Multi-Provider Support**: OpenAI, Ollama, Gemini, Perplexity
- **Dual Modes**: Text generation and image analysis
- **CLI & Web UI**: Command-line and Streamlit interfaces
- **Comprehensive Error Handling**: Multi-layered exception handling with logging
- **Developer-Friendly**: Excellent tooling via Makefile

### Core Metrics
- **Total Lines**: ~822 lines of Python
- **Test Coverage**: 21 unit tests with comprehensive mocking
- **Dependencies**: 5 core packages (litellm, streamlit, pytest, python-dotenv, Pillow)
- **Modules**: 4 core modules in `lite/`, 2 application scripts

---

## Architecture & Design Patterns

### Design Philosophy
The codebase follows **composition over inheritance** with a focus on:
- Single Responsibility Principle
- Separation of Concerns
- DRY (Don't Repeat Yourself)
- Intelligent Defaults
- Graceful Degradation

### Core Architecture Pattern: Unified Client

```
┌─────────────────────────────────────────────┐
│  CLI/Web Interface (scripts/)               │
│  - liteclient_cli.py                        │
│  - streamlit_liteclient.py                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  LiteClient (lite/lite_client.py)           │
│  - generate_text()                          │
│  - create_message()                         │
│  - list_models() / get_model()              │
└──┬─────────────┬────────────────┬───────────┘
   │             │                │
   ▼             ▼                ▼
┌────────┐  ┌──────────┐  ┌──────────────┐
│ Model  │  │  Image   │  │   Logging    │
│ Config │  │  Utils   │  │   Config     │
└────────┘  └──────────┘  └──────────────┘
   │             │
   └─────┬───────┘
         ▼
    litellm.completion()
```

### Key Classes & Responsibilities

#### 1. **ModelConfig** (`lite/config.py`)
**Type**: Dataclass with class methods
**Responsibility**: Centralized model configuration and retrieval

```python
@dataclass
class ModelConfig:
    OPENAI_TEXT_MODELS = ["openai/gpt-4o", "openai/gpt-4o-mini"]
    OLLAMA_TEXT_MODELS = ["ollama/llama3.2", "ollama/phi4"]
    GEMINI_TEXT_MODELS = ["gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash-lite"]
    PERPLEXITY_TEXT_MODELS = ["perplexity/sonar", "perplexity/sonar-pro"]

    @classmethod
    def get_model(cls, index: int, model_type: str = "text") -> Optional[str]

    @classmethod
    def get_models(cls, model_type: str = "text") -> List[str]
```

**Key Features**:
- Index-based model selection (0-N)
- Type-based filtering (text vs vision)
- Validation on retrieval
- Returns `None` for invalid indexes

#### 2. **ImageUtils** (`lite/image_utils.py`)
**Type**: Static utility class
**Responsibility**: Image validation and base64 encoding

```python
class ImageUtils:
    @staticmethod
    def encode_to_base64(image_path: str) -> str

    @staticmethod
    def is_valid_image(path: Path) -> bool
```

**Supported Formats**: JPG, JPEG, PNG, GIF, WebP
**Error Handling**: FileNotFoundError, ValueError with detailed logging

#### 3. **LiteClient** (`lite/lite_client.py`)
**Type**: Main client class
**Responsibility**: Unified API for text/vision interactions

```python
class LiteClient:
    DEFAULT_TEMPERATURE = 0.2

    def generate_text(
        self,
        prompt: str,
        model: str,
        image_path: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Union[str, Dict[str, Any]]

    @staticmethod
    def create_message(prompt: str, image_path: Optional[str] = None) -> list

    @staticmethod
    def list_models(model_type: str = "text") -> list

    @staticmethod
    def get_model(index: int, model_type: str = "text") -> Optional[str]
```

**Key Design Decisions**:
- Single method (`generate_text`) handles both text and vision
- Return type varies by context: `str` for text, `Dict[str, Any]` for vision errors
- Static methods for utility functions
- Intelligent defaults (empty prompt → "Describe the image" for vision)

---

## Codebase Structure

### Directory Layout

```
LiteLLM/
├── lite/                          # Core library package (346 lines)
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Model configuration (58 lines)
│   ├── image_utils.py             # Image processing (62 lines)
│   ├── logging_config.py          # Logging setup (60 lines)
│   └── lite_client.py             # Main client (166 lines)
│
├── scripts/                       # Application scripts (259 lines)
│   ├── liteclient_cli.py          # CLI interface (133 lines)
│   └── streamlit_liteclient.py    # Web UI (126 lines)
│
├── tests/                         # Test suite (217 lines)
│   ├── __init__.py                # Package marker
│   └── test_litetext.py           # Unit tests (216 lines)
│
├── examples/                      # Usage examples
│   ├── basic/                     # Simple usage examples
│   │   └── gemini_chat.py
│   ├── structured_output/         # Pydantic/JSON examples
│   │   ├── drugbank_medicine.py
│   │   ├── jsonout.py
│   │   ├── medicine_info.py
│   │   └── nobel_prize_info.py
│   ├── web_search/                # Web search examples
│   │   ├── google_search.py
│   │   ├── url_explain.py
│   │   └── websearch.py
│   ├── advanced/                  # Advanced patterns
│   │   └── perplx_chat.py
│   ├── README.md                  # Examples documentation
│   └── simple_example.py          # Basic litellm usage
│
├── Makefile                       # Development automation (114 lines)
├── setup.py                       # Package configuration (27 lines)
├── requirements.txt               # Dependencies (6 lines)
├── README.md                      # User documentation (230 lines)
├── CLAUDE.md                      # AI assistant guide (this file)
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore rules
```

### Key Files & Their Purposes

#### Core Library (`lite/`)

| File | Purpose | Key Exports |
|------|---------|-------------|
| `config.py` | Model configuration and constants | `ModelConfig`, `DEFAULT_TEMPERATURE`, `SUPPORTED_IMAGE_TYPES` |
| `image_utils.py` | Image validation and encoding | `ImageUtils` |
| `logging_config.py` | Centralized logging configuration | `setup_logging()` |
| `lite_client.py` | Main client implementation | `LiteClient` |

**Note**: The `lite/` directory now has an `__init__.py` file that exports `LiteClient` and `ModelConfig` for convenient package-level imports.

#### Application Scripts (`scripts/`)

| File | Purpose | Entry Point |
|------|---------|-------------|
| `liteclient_cli.py` | Command-line interface | `cli_interface()` |
| `streamlit_liteclient.py` | Web UI interface | Streamlit app |

**Import Pattern**: Scripts add parent directory to `sys.path`:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from lite.lite_client import LiteClient
```

#### Configuration Files

| File | Purpose |
|------|---------|
| `Makefile` | Development workflow automation (venv, install, test, lint, format, run, clean) |
| `setup.py` | Package metadata and dependencies for `pip install` |
| `requirements.txt` | Core dependencies with version constraints |

---

## Development Workflows

### Initial Setup

```bash
# 1. Create virtual environment
make venv

# 2. Activate virtual environment
source litenv/bin/activate

# 3. Install dependencies (development mode)
make install-dev

# 4. Verify installation
make test
```

### Daily Development Workflow

```bash
# 1. Activate virtual environment
source litenv/bin/activate

# 2. Run tests before making changes
make test

# 3. Make your changes to code

# 4. Format code
make format

# 5. Run linter
make lint

# 6. Run tests with coverage
make test

# 7. Run the application
make run-cli-text     # Test CLI
make run-web-text     # Test Streamlit UI
```

### Makefile Commands Reference

#### Virtual Environment
- `make venv` - Create virtual environment (litenv/)
- `make venv-activate` - Show activation instructions
- `make clean-venv` - Remove virtual environment

#### Installation
- `make install` - Install core dependencies
- `make install-dev` - Install with dev tools (pylint, black, pytest-cov)

#### Testing & Quality
- `make test` - Run pytest with coverage (HTML report)
- `make test-watch` - Run pytest with short traceback
- `make lint` - Run pylint on lite/ and scripts/
- `make format` - Format with black (lite/, scripts/, tests/)

#### Running Applications
- `make run-cli-text` - Run CLI (list text models)
- `make run-cli-vision` - Run CLI (show help)
- `make run-web-text` - Run Streamlit text UI
- `make run-web-vision` - Run Streamlit vision UI

#### Cleanup
- `make clean` - Remove cache, build files, __pycache__
- `make clean-all` - Full cleanup (cache + venv)

### Testing Workflow

```bash
# Run all tests with coverage
make test

# Run specific test file
litenv/bin/python -m pytest tests/test_litetext.py -v

# Run specific test class
litenv/bin/python -m pytest tests/test_litetext.py::TestLiteClient -v

# Run specific test method
litenv/bin/python -m pytest tests/test_litetext.py::TestLiteClient::test_generate_text_success -v

# Run tests with print statements
litenv/bin/python -m pytest tests/ -v -s

# View coverage report
# Opens htmlcov/index.html in browser after `make test`
```

---

## Coding Conventions

### Import Organization

**Standard Pattern** (strictly follow this order):
```python
# 1. Standard library imports
import argparse
import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

# 2. Third-party imports
from litellm import completion, APIError
import streamlit as st

# 3. Local imports
from config import ModelConfig
from image_utils import ImageUtils
```

### Type Hints

**Always use type hints** for function signatures:

```python
# Good
def generate_text(
    self,
    prompt: str,
    model: str,
    image_path: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Union[str, Dict[str, Any]]:
    """Generate text or analyze image."""
    pass

# Common type patterns
from typing import Any, Dict, List, Optional, Union

# Return types
-> str                          # Simple string
-> Optional[str]                # String or None
-> Union[str, Dict[str, Any]]   # String or dict
-> List[str]                    # List of strings
-> Dict[str, Any]               # Dictionary
```

### Docstrings

**Use Google-style docstrings**:

```python
def generate_text(self, prompt: str, model: str, ...) -> Union[str, Dict[str, Any]]:
    """
    Generate text from a prompt or analyze an image with a prompt.

    Args:
        prompt: The input prompt for the model or image analysis
        model: The model identifier (e.g., "openai/gpt-4o")
        image_path: Optional path to image file for vision analysis
        temperature: Sampling temperature (default: 0.2)

    Returns:
        Generated text response as string or error dict

    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If image is invalid format
        APIError: If API call fails
    """
```

**Required sections**:
- One-line summary
- `Args:` - All parameters with descriptions
- `Returns:` - What the function returns
- `Raises:` (if applicable) - Exceptions that can be raised

### Naming Conventions

```python
# Constants (uppercase with underscores)
DEFAULT_TEMPERATURE = 0.2
SUPPORTED_IMAGE_TYPES = ("jpg", "jpeg", "png", "gif", "webp")

# Classes (PascalCase)
class LiteClient:
class ModelConfig:
class ImageUtils:

# Functions and methods (snake_case)
def generate_text():
def create_message():
def encode_to_base64():

# Private methods (leading underscore)
def _internal_helper():

# Module-level loggers
logger = logging.getLogger(__name__)
```

### Class Design Patterns

```python
# 1. Dataclasses for configuration
from dataclasses import dataclass

@dataclass
class ModelConfig:
    OPENAI_TEXT_MODELS = ["openai/gpt-4o"]

    @classmethod
    def get_model(cls, index: int) -> Optional[str]:
        pass

# 2. Static methods for utilities
class ImageUtils:
    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        pass

# 3. Instance methods for stateful operations
class LiteClient:
    DEFAULT_TEMPERATURE = 0.2

    def generate_text(self, prompt: str) -> str:
        pass
```

### Constants and Configuration

**Define at module level**:
```python
# At top of config.py
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000
DEFAULT_PROMPT = "Describe this image in detail"
SUPPORTED_IMAGE_TYPES = ("jpg", "jpeg", "png", "gif", "webp")
IMAGE_MIME_TYPE = "image/jpeg"
```

**Import and use**:
```python
from config import DEFAULT_TEMPERATURE, SUPPORTED_IMAGE_TYPES
```

---

## Error Handling Patterns

### Comprehensive Exception Handling

**Always use specific exceptions first, general last**:

```python
def generate_text(self, prompt: str, model: str, ...) -> Union[str, Dict[str, Any]]:
    """Generate text with comprehensive error handling."""

    try:
        # Main logic
        messages = self.create_message(prompt, image_path)
        response = completion(model=model, messages=messages, temperature=temperature)
        return response.choices[0].message.content

    except FileNotFoundError as e:
        # Specific error: file operations
        error_msg = f"File error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg} if image_path else error_msg

    except ValueError as e:
        # Specific error: validation
        error_msg = f"Validation Error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg} if image_path else error_msg

    except APIError as e:
        # Specific error: API failures
        error_msg = f"API Error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg} if image_path else error_msg

    except Exception as e:
        # Catch-all for unexpected errors
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg} if image_path else error_msg
```

### Input Validation Pattern

**Validate early, provide intelligent defaults**:

```python
def generate_text(self, prompt: str, ...) -> Union[str, Dict[str, Any]]:
    """Validate inputs with intelligent defaults."""

    # Validate prompt with context-aware defaults
    if not prompt or not prompt.strip():
        if image_path:
            prompt = "Describe the image"  # Intelligent default for vision
        else:
            return "Error: Prompt cannot be empty"  # Error for text mode

    # Continue with validated inputs
    ...
```

### Logging Pattern

**Always log at key decision points**:

```python
import logging

logger = logging.getLogger(__name__)

def generate_text(self, prompt: str, model: str, ...) -> Union[str, Dict[str, Any]]:
    """Log request lifecycle."""

    # Log request initiation with context
    log_action = "Analyzing image" if image_path else "Generating text"
    logger.info(f"{log_action} with model: {model}")

    try:
        # Main logic
        response = completion(...)

        # Log success
        logger.info("Request successful")
        return response.choices[0].message.content

    except Exception as e:
        # Log errors with context
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
```

**Logging Levels**:
- `logger.info()` - Successful operations, request tracking
- `logger.error()` - All error conditions with context
- `logger.debug()` - Detailed debugging info (not currently used)

---

## Testing Guidelines

### Test Structure

**Organization** (`tests/test_litetext.py`):
```python
import sys
import os
from unittest.mock import patch, MagicMock

# Add lite/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lite'))

class TestModelConfig:
    """Test ModelConfig class."""

    def test_get_model_valid_index_text(self):
        """Test valid text model retrieval."""
        pass

    # 7 total tests

class TestLiteClient:
    """Test LiteClient class."""

    def test_generate_text_success(self, mock_completion):
        """Test successful text generation."""
        pass

    # 14 total tests
```

### Naming Convention

**Pattern**: `test_<method>_<scenario>_<expected>()`

```python
# Good examples
def test_get_model_valid_index_text():
def test_get_model_invalid_index_returns_none():
def test_generate_text_empty_prompt_no_image():
def test_create_message_with_image():
def test_generate_text_api_error():

# Bad examples
def test_model():              # Too vague
def test_everything():         # Not specific
def test_generate_text():      # Missing scenario
```

### Mocking Strategy

**Mock external dependencies, not internal logic**:

```python
@patch('lite_client.completion')  # Mock litellm API
def test_generate_text_success(self, mock_completion):
    """Test successful generation with mocked API."""

    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Test response"
    mock_completion.return_value = mock_response

    # Execute
    client = LiteClient()
    result = client.generate_text(
        prompt="Test prompt",
        model="openai/gpt-4o",
        temperature=0.5
    )

    # Verify
    assert result == "Test response"
    mock_completion.assert_called_once()
```

**Components to Mock**:
- `litellm.completion()` - All API calls
- `ImageUtils.encode_to_base64()` - File I/O operations
- Response objects using `MagicMock()`

**DO NOT Mock**:
- Internal class methods (test them directly)
- Simple utility functions
- Configuration classes

### Test Categories

1. **Configuration Tests** (`TestModelConfig`):
   - Valid/invalid indexes
   - Model type selection
   - Boundary conditions

2. **Unit Tests** (`TestLiteClient`):
   - Message creation
   - Input validation
   - Return type verification

3. **Integration Tests** (mocked):
   - End-to-end flow
   - Parameter passing
   - Error propagation

4. **Error Handling Tests**:
   - Empty/whitespace prompts
   - File not found
   - API errors
   - Unexpected exceptions

### Writing New Tests

**Template**:
```python
class TestNewFeature:
    """Test new feature functionality."""

    @patch('lite_client.completion')
    def test_new_feature_success(self, mock_completion):
        """Test new feature with successful execution."""
        # Arrange
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Expected output"
        mock_completion.return_value = mock_response

        # Act
        client = LiteClient()
        result = client.new_method(param="value")

        # Assert
        assert result == "Expected output"
        mock_completion.assert_called_once()

    def test_new_feature_validation_error(self):
        """Test new feature with invalid input."""
        # Act
        client = LiteClient()
        result = client.new_method(param="")

        # Assert
        assert "Error:" in result
```

---

## Common Tasks

### Adding a New Model Provider

**Steps**:

1. **Update `lite/config.py`**:
```python
@dataclass
class ModelConfig:
    # Add new provider models
    NEWPROVIDER_TEXT_MODELS = ["newprovider/model-1", "newprovider/model-2"]

    # Update TEXT_MODELS list
    TEXT_MODELS = (
        OPENAI_TEXT_MODELS +
        OLLAMA_TEXT_MODELS +
        GEMINI_TEXT_MODELS +
        PERPLEXITY_TEXT_MODELS +
        NEWPROVIDER_TEXT_MODELS  # Add here
    )
```

2. **Update README.md**:
```markdown
### Text Models
- NewProvider: `newprovider/model-1`, `newprovider/model-2`
```

3. **Add tests** in `tests/test_litetext.py`:
```python
def test_get_model_newprovider():
    """Test NewProvider model retrieval."""
    model = ModelConfig.get_model(10, "text")  # Adjust index
    assert "newprovider/" in model
```

4. **Set environment variable** (if needed):
```bash
export NEWPROVIDER_API_KEY="your-api-key"
```

5. **Test**:
```bash
make test
python scripts/liteclient_cli.py --list-models
```

### Adding a New Feature to LiteClient

**Example**: Add support for max_tokens parameter

1. **Update `lite/lite_client.py`**:
```python
class LiteClient:
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 2000  # Add constant

    def generate_text(
        self,
        prompt: str,
        model: str,
        image_path: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,  # Add parameter
    ) -> Union[str, Dict[str, Any]]:
        """Generate text with max_tokens support."""

        try:
            messages = self.create_message(prompt, image_path)
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,  # Pass to API
            )
            return response.choices[0].message.content
        except Exception as e:
            # Error handling...
```

2. **Update `scripts/liteclient_cli.py`**:
```python
parser.add_argument(
    "--max-tokens",
    type=int,
    default=LiteClient.DEFAULT_MAX_TOKENS,
    help=f"Maximum tokens to generate (default: {LiteClient.DEFAULT_MAX_TOKENS})",
)

# In execution
result = client.generate_text(
    prompt=args.question,
    model=model,
    temperature=args.temperature,
    max_tokens=args.max_tokens,  # Add parameter
)
```

3. **Add tests**:
```python
@patch('lite_client.completion')
def test_generate_text_with_max_tokens(self, mock_completion):
    """Test generation with custom max_tokens."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Response"
    mock_completion.return_value = mock_response

    client = LiteClient()
    result = client.generate_text(
        prompt="Test",
        model="openai/gpt-4o",
        max_tokens=1000
    )

    # Verify max_tokens was passed
    call_args = mock_completion.call_args
    assert call_args[1]['max_tokens'] == 1000
```

4. **Update documentation** in `README.md`

### Debugging Failed Tests

**Strategy**:
```bash
# 1. Run single test with verbose output
litenv/bin/python -m pytest tests/test_litetext.py::TestLiteClient::test_name -v -s

# 2. Check test output for failures
# Look for assertion errors, mock issues, import errors

# 3. Add print statements temporarily
def test_failing_case(self):
    result = client.generate_text(...)
    print(f"DEBUG: Result = {result}")  # Temporary debug
    assert result == "expected"

# 4. Check mock setup
@patch('lite_client.completion')
def test_case(self, mock_completion):
    # Verify mock is configured correctly
    print(f"Mock called: {mock_completion.called}")
    print(f"Call args: {mock_completion.call_args}")

# 5. Run with coverage to see what's not covered
make test
# Open htmlcov/index.html to see coverage gaps
```

### Adding Logging to New Code

**Pattern**:
```python
import logging

logger = logging.getLogger(__name__)

def new_function(param: str) -> str:
    """New function with logging."""

    # Log entry with context
    logger.info(f"Processing {param}")

    try:
        # Main logic
        result = do_something(param)

        # Log success
        logger.info(f"Successfully processed {param}")
        return result

    except Exception as e:
        # Log error with full context
        logger.error(f"Failed to process {param}: {str(e)}")
        raise
```

**Log Configuration** (already set up in `lite/logging_config.py`):
- File: `litellm.log`
- Format: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- Level: INFO (configurable)

---

## Git Workflow

### Branch Strategy

**Current Branch**: `claude/claude-md-milxo9w3t7tmuz8e-01TmstPFLEGHUt5rudyd7zEN`

**Branch Naming Convention**:
- Feature branches: `claude/<feature-name>-<session-id>`
- CRITICAL: Branch must start with `claude/` and end with session ID for push to work

### Commit Message Format

**Pattern**: Follow the existing style from git history

```bash
# Good examples (from git log)
"Update supported models: Gemini 2.5 and Perplexity"
"Consolidate text and vision tools into unified LiteClient interface"
"Fix capitalization in error message"
"Update Makefile lint target to match new directory structure"
"Add virtual environment support to Makefile"

# Pattern: <Action> <subject>: <details>
# Actions: Add, Update, Fix, Refactor, Remove, Consolidate, Restructure
```

**Guidelines**:
- Start with imperative verb (Add, Update, Fix, etc.)
- Be specific about what changed
- Keep under 72 characters
- No period at the end
- Use present tense

### Making Commits

```bash
# 1. Check status
git status

# 2. Review changes
git diff

# 3. Stage changes
git add lite/config.py
git add tests/test_litetext.py

# 4. Commit with descriptive message
git commit -m "Add NewProvider support to ModelConfig"

# 5. Push to feature branch
git push -u origin claude/claude-md-milxo9w3t7tmuz8e-01TmstPFLEGHUt5rudyd7zEN
```

### Push Retry Strategy

**CRITICAL**: If push fails due to network errors, retry up to 4 times with exponential backoff:

```bash
# Retry logic (2s, 4s, 8s, 16s delays)
git push -u origin <branch-name> || sleep 2 && git push -u origin <branch-name> || sleep 4 && git push -u origin <branch-name> || sleep 8 && git push -u origin <branch-name>
```

### Git Safety

**NEVER**:
- Push to main/master without permission
- Use `git push --force` without explicit user request
- Skip hooks (`--no-verify`, `--no-gpg-sign`)
- Amend commits from other developers
- Commit sensitive data (.env files, API keys)

**ALWAYS**:
- Review `git status` before committing
- Review `git diff` to verify changes
- Use `-u` flag on first push: `git push -u origin <branch-name>`
- Follow the branch naming convention (starts with `claude/`)

---

## Key Patterns to Follow

### 1. Intelligent Defaults

**Provide sensible defaults based on context**:

```python
# Good: Context-aware defaults
def generate_text(self, prompt: str, image_path: Optional[str] = None):
    if not prompt or not prompt.strip():
        if image_path:
            prompt = "Describe the image"  # Default for vision mode
        else:
            return "Error: Prompt cannot be empty"  # Error for text mode
```

### 2. Graceful Degradation

**Handle errors without crashing**:

```python
# Good: Return error messages instead of raising
try:
    result = api_call()
    return result
except APIError as e:
    return {"error": f"API Error: {str(e)}"}  # Graceful failure
```

### 3. Single Responsibility

**Each module/class has one clear purpose**:

```python
# Good: Single responsibility
class ImageUtils:
    """Handles ONLY image operations."""

    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        pass

    @staticmethod
    def is_valid_image(path: Path) -> bool:
        pass
```

### 4. Separation of Concerns

**Keep layers independent**:

```
CLI Layer (scripts/)           → User interface
    ↓
Business Logic (lite_client)   → Core functionality
    ↓
Utilities (config, image)      → Shared utilities
    ↓
External APIs (litellm)        → Third-party services
```

### 5. DRY (Don't Repeat Yourself)

**Centralize shared code**:

```python
# Good: Centralized model configuration
class ModelConfig:
    TEXT_MODELS = OPENAI_TEXT_MODELS + OLLAMA_TEXT_MODELS + ...

    @classmethod
    def get_model(cls, index: int, model_type: str):
        """Single source of truth for model retrieval."""
        pass

# Good: Reuse via static methods
@staticmethod
def list_models(model_type: str = "text") -> list:
    return ModelConfig.get_models(model_type=model_type)
```

### 6. Type Safety

**Use type hints everywhere**:

```python
# Good: Clear type contracts
def generate_text(
    self,
    prompt: str,
    model: str,
    image_path: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Union[str, Dict[str, Any]]:
    pass
```

### 7. Path Handling

**Use pathlib for file operations**:

```python
from pathlib import Path

# Good: Use Path objects
path = Path(image_path)
if not path.exists():
    raise FileNotFoundError(f"Image file not found: {image_path}")

if path.suffix.lower() not in valid_extensions:
    raise ValueError(f"Invalid image format")
```

### 8. Comprehensive Logging

**Log at key decision points**:

```python
# Good: Log request lifecycle
logger.info(f"Starting operation with model: {model}")
try:
    result = operation()
    logger.info("Operation successful")
    return result
except Exception as e:
    logger.error(f"Operation failed: {str(e)}")
    raise
```

---

## Environment & Dependencies

### Required Environment Variables

```bash
# Required for respective providers
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export PERPLEXITY_API_KEY="..."

# Optional: Load from .env file
# Create .env file in project root with above variables
```

**Note**: Ollama requires local installation, no API key needed.

### Dependencies (`requirements.txt`)

```
litellm>=1.0.0          # Core LLM library (unified API for multiple providers)
streamlit>=1.28.0       # Web UI framework
pytest>=7.4.0           # Testing framework
python-dotenv>=1.0.0    # Environment variable management
Pillow>=10.0.0          # Image processing
```

**Installation**:
```bash
pip install -r requirements.txt
```

### Development Dependencies

**Installed via `make install-dev`**:
```bash
pylint              # Code linting
black               # Code formatting
pytest-cov          # Test coverage reporting
```

### Python Version Requirements

- **Minimum**: Python 3.8
- **Recommended**: Python 3.12 (as specified in Makefile)
- **Check version**: `python --version`

### Virtual Environment

**Location**: `litenv/` (created by `make venv`)

**Activation**:
```bash
# Linux/macOS
source litenv/bin/activate

# Windows
litenv\Scripts\activate
```

**Deactivation**:
```bash
deactivate
```

---

## Quick Reference

### Most Common Commands

```bash
# Setup (first time)
make venv && source litenv/bin/activate && make install-dev

# Daily workflow
make format && make lint && make test

# Run applications
python scripts/liteclient_cli.py -q "What is AI?"
python scripts/liteclient_cli.py -i photo.jpg
streamlit run scripts/streamlit_liteclient.py

# Testing
make test                    # Full suite with coverage
make test-watch              # Quick run
litenv/bin/python -m pytest tests/test_litetext.py::TestClass::test_method -v

# Cleanup
make clean                   # Remove cache
make clean-all               # Remove cache + venv
```

### File Templates

**New Test Class**:
```python
class TestNewFeature:
    """Test new feature functionality."""

    @patch('lite_client.completion')
    def test_new_feature_success(self, mock_completion):
        """Test successful execution."""
        # Arrange
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Expected"
        mock_completion.return_value = mock_response

        # Act
        client = LiteClient()
        result = client.new_method(param="value")

        # Assert
        assert result == "Expected"
```

**New Utility Class**:
```python
"""Module for new utility functionality."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class NewUtils:
    """Utility class for new feature."""

    @staticmethod
    def utility_method(param: str) -> str:
        """
        Utility method description.

        Args:
            param: Description of parameter

        Returns:
            Description of return value

        Raises:
            ValueError: When parameter is invalid
        """
        if not param:
            raise ValueError("Parameter cannot be empty")

        logger.info(f"Processing: {param}")
        return f"Processed: {param}"
```

---

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `sys.path` includes parent directory in scripts
2. **API errors**: Check environment variables are set correctly
3. **Test failures**: Verify mocks are configured for all external calls
4. **Push failures**: Ensure branch starts with `claude/` and ends with session ID
5. **Missing `__init__.py`**: Add to `lite/` if package import issues occur

### Getting Help

- **README.md**: User-facing documentation
- **This file (CLAUDE.md)**: Developer/AI assistant guide
- **Git history**: `git log --oneline` for recent changes
- **Tests**: `tests/test_litetext.py` for usage examples

---

## Summary

LiteLLM is a well-architected Python project providing unified access to multiple LLM providers. Key principles:

1. **Unified Interface**: Single client for text and vision
2. **Clean Architecture**: Separation of concerns, single responsibility
3. **Comprehensive Testing**: 21 unit tests with mocking
4. **Developer Experience**: Excellent Makefile, clear conventions
5. **Error Resilience**: Multi-layered error handling with logging
6. **Type Safety**: Extensive type hints throughout

When contributing, follow the patterns established in the codebase, write tests for new features, and maintain the clean separation between core logic, utilities, and applications.

---

**Generated**: 2025-11-30
**For**: AI Assistants working with LiteLLM codebase
