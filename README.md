# Seshat: LLMOps System for LLM Developers and Prompt Engineers

## Overview

Seshat is an advanced LLMOps (Large Language Model Operations) system designed to streamline and enhance the workflow for LLM developers and prompt engineers. This project leverages a modular interface to manage various components such as batch processing, embeddings, meta-prompts, FAQ summaries, and system configurations. Seshat aims to provide a comprehensive and user-friendly environment for prompts and data processing with LLM.

## Features

- **Batch Processing**: Efficiently handle large batches of data with LLM.
![Batch LLM](screenshots/Seshat_Batch.png?raw=true)
![Batch LLM](screenshots/Seshat_Batch_1.png?raw=true)
- **Lang Serve Client**: Interface for interacting with [LangServe](https://python.langchain.com/v0.2/docs/langserve/) applications.
- **Embedding & VDB**: Embed strings/sheets, import embeddings to vector databases, and cluster embeddings.
![Embedding](screenshots/Sechat_Embedding.png?raw=true)
![Clustering](screenshots/Seshat_Clustering.png?raw=true)
- **Meta Prompt Management**: Create prompts with LLM, based on [meta-prompt](https://github.com/yaleh/meta-prompt).
[![Meta Prompt](https://img.youtube.com/vi/eNFUq2AjKCk/0.jpg)](https://www.youtube.com/watch?v=eNFUq2AjKCk)
- **FAQ Summary Fix**: Preprocess FAQ data, convert Q&A sheets into `faq_text` and `rdb_text`. `faq_text` is used by VDB and LLM for inference, while `rdb_text` is used for output.
- **System Configuration**: Comprehensive system management and configuration, including updating models of OpenRouter.

## Installation

To install Seshat, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/seshat.git
cd seshat
poetry install
```

## Configuration

Seshat uses a configuration file (`config.yaml`) for setting up various parameters. You can specify the path to your configuration file using the `--config_file` argument:

```bash
python main.py --config_file path/to/your/config.yaml
```

### API Keys

Some API keys are required to use certain features of Seshat. You can set these keys in the configuration file:

- embedding.text-embedding-3-small.openai_api_key
- embedding.azure-text-embedding-3-small.openai_api_key
- llm.llm_services.OpenAI.args.openai_api_key
- llm.llm_services.OpenRouter.args.openai_api_key
- llm.llm_services.Cloudflare.args.openai_api_key
- llm.llm_services.Azure_OpenAI.args.azure_openai_api_key
- llm.llm_services.Replicate.args.replicate_api_key
- llm.llm_services.HuggingFace.args.huggingface_api_key

## Usage

To launch the Seshat application, run the following command:

```bash
python main.py
```

By default, the application will look for a `config.yaml` file in the current directory. You can customize the configuration by modifying this file or by providing a different configuration file.

## Docker

Docker image is also available for Seshat. To run Seshat in a Docker container, use the following command:

```bash
docker run -p 7860:7860 -v /path/to/your/config.yaml:/app/config.yaml ghcr.io/yaleh/seshat:main
```

## User Interface

Seshat provides a tabbed interface with the following sections:

- **Batch**: Batch processing management.
- **Lang Serve**: Language server interaction.
- **Embedding & VDB**: Embeddings and vector database management.
- **Meta Prompt**: Meta-prompt creation and management.
- **FAQ Summary Fix**: FAQ summarization and fixing tools.
- **System**: System configuration and management.

## Development

To contribute to Seshat, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with a descriptive message.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

## Dependencies

Poetry is used for managing dependencies in this project. To install the dependencies, run the following command:

```
poetry install
```

You can create a new virtual environment using Poetry by running:

```
python -m venv venv
source venv/bin/activate
pip install -U poetry
poetry install
```

## Known Issues

* Gradio is known to freeze with multiple concurrent requests. If this happens, try restarting the service.
    * If you are using a Docker instance with the `restart` policy set to `always`, you can click the `Exit` button on the `System` tab to restart the service.

## License

Seshat is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

We would like to thank the contributors and the open-source community for their invaluable support and contributions to this project.

## Contact

For any questions or inquiries, please contact the project maintainer at [calvino.huang@gmail.com].