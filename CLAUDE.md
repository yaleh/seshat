# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
# Install dependencies
poetry install

# Run application
python seshat.py
# or with custom config
python seshat.py --config_file path/to/config.yaml

# Run via Poetry
poetry run python seshat.py
```

### Testing
```bash
# Run all tests
python -m unittest discover test/

# Run specific test
python test/table_parser_test.py

# With Poetry
poetry run python -m unittest test.table_parser_test
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
- **Python 3.10-3.12** with **Poetry** dependency management
- **Gradio 4.26.0** for web UI
- **LangChain ecosystem** for LLM integration
- **Pydantic** for configuration validation
- **SQLite** for message persistence
- **Docker** for containerization