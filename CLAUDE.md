# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup
```bash
# Ensure correct Python version (requires 3.10-3.12)
pyenv local 3.10.18

# Configure Poetry to use correct Python version
poetry env use $(which python3.10)

# Install dependencies
poetry install
```

### Development
```bash
# Run application
poetry run python seshat.py
# or with custom config
poetry run python seshat.py --config_file path/to/config.yaml
```

### Testing
```bash
# Run all tests individually
poetry run python test/table_parser_test.py      # Table parsing (9 tests)
poetry run python test/test_config_loader.py     # Configuration (18 tests) 
poetry run python test/test_lcel.py              # LLM integration (12 tests)
poetry run python test/test_db_sqlite3.py        # Database (14 tests)
poetry run python test/test_utils.py             # Utilities (8 tests)
poetry run python test/test_dataframe_io.py      # DataFrame I/O (13 tests)
poetry run python test/test_chatbot.py           # Chatbot (13 tests)

# Run all tests with coverage
poetry run coverage run test/table_parser_test.py
poetry run coverage run -a test/test_config_loader.py
poetry run coverage run -a test/test_lcel.py
poetry run coverage run -a test/test_db_sqlite3.py
poetry run coverage run -a test/test_utils.py
poetry run coverage run -a test/test_dataframe_io.py
poetry run coverage run -a test/test_chatbot.py

# Generate coverage reports
poetry run coverage report          # Terminal report
poetry run coverage html            # HTML report in htmlcov/

# Current coverage: 19.07% overall
# Modules with 100% coverage: config_loader, lcel, db_sqlite3, utils, dataframe_io
# High coverage: chatbot (75.79%), table_parser (92.31%)
```

### Docker
```bash
# Build image
docker build -t seshat .

# Run container
docker run -p 7860:7860 -v /path/to/config.yaml:/app/config.yaml seshat

# Use pre-built image
docker run -p 7860:7860 -v /path/to/config.yaml:/app/config.yaml ghcr.io/yaleh/seshat:main
```

## Architecture

### Core Structure
Seshat is an LLMOps platform built with **Gradio** and **LangChain**, featuring a modular tabbed interface for various AI workflows.

**Key Components:**
- `seshat.py` - Main application entry point that orchestrates the Gradio interface
- `components/` - Core business logic (chatbot, LLM model factory)
- `ui/` - Modular Gradio UI components, each implementing a specific workflow tab
- `tools/` - Utilities for configuration, parsing, and data I/O
- `db/` - SQLite database layer for message storage

### UI Module Pattern
Each UI module follows the same pattern:
- Inherits from base UI class
- Implements `create_ui()` method returning Gradio components
- Handles specific workflow (batch processing, embeddings, meta-prompts, etc.)
- Example: `BatchUI`, `EmbeddingUI`, `MetaPromptUI`

### Configuration System
Uses **Pydantic** models for type-safe configuration management:
- `config.yaml` - Main configuration file
- `ConfigLoader` - Pydantic-based configuration validation
- Supports multiple LLM providers: OpenAI, OpenRouter, Azure, Cloudflare, HuggingFace
- Vector database configurations: Pinecone, Milvus, Chroma, FAISS

### LLM Integration
- `LLMModelFactory` - Factory pattern for dynamic model instantiation
- LangChain-based with support for multiple providers
- Unified interface across different AI services
- Configuration-driven model selection

### Data Processing
- `TableParser` - Markdown table parsing with comprehensive error handling
- `DataFrameIO` - Import/export utilities for various data formats
- Batch processing capabilities for large datasets
- Vector embedding and clustering workflows

## Key Technologies
- **Python 3.10-3.12** with **Poetry** dependency management and **pyenv** for version management
- **Gradio 4.26.0** for web UI
- **LangChain ecosystem** for LLM integration
- **Pydantic** for configuration validation
- **SQLite** for message persistence
- **Docker** for containerization

## Important Notes
- **Python Version**: Project requires Python 3.10-3.12. Use `pyenv local 3.10.18` to set the correct version.
- **Poetry Environment**: Always configure Poetry to use the correct Python version with `poetry env use $(which python3.10)` before installing dependencies.
- **Testing**: The project uses Python's built-in `unittest` framework. Run tests via Poetry to ensure correct environment.