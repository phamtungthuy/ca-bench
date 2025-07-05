# CA-Bench

CA-Bench is a comprehensive benchmark suite for evaluating the capabilities of large language models (LLMs) and multimodal models across a variety of tasks and domains. It provides a standardized framework for running, evaluating, and comparing models using reproducible workflows and metrics.

## Features

- **Task Suite**: Includes a diverse set of tasks at node, chain, and graph levels.
- **Automated Evaluation**: Supports multiple evaluation metrics (accuracy, BLEU, CodeBLEU, F1, R2, semantic similarity, etc.).
- **Workflow Generation**: Automatically generates and runs workflows for each task.
- **Extensible**: Easily add new tasks, models, and evaluation metrics.
- **Cost Tracking**: Tracks API usage and cost for supported providers (e.g., OpenAI).
- **Multimodal Support**: Handles text, image, audio, and video tasks.

## Directory Structure

```text
├── configs/ # Configuration files for models and tasks
├── data/ # Model metadata
├── provider/ # LLM provider integrations (OpenAI, etc.)
├── scripts/ # Main scripts for evaluation, data download, etc.
├── server/ # API server for model inference
├── templates/ # Prompt and workflow templates
├── utils/ # Utility modules (token counting, cost management, etc.)
├── Dockerfile # Docker support
├── requirements.txt # Python dependencies
└── README.md # This file
```

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) Docker

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-org/ca-bench.git
   cd ca-bench
   ```

2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. (Optional) Download benchmark datasets:
   ```sh
   python scripts/download_data.py --datasets tasks
   ```

### Running Evaluations

To evaluate models on a specific range and task:

```sh
python evaluate.py --range task --result_folder_name results --folder_name <task_folder>
```

For all available options, run:

```sh
python evaluate.py --help
```

Using Docker
Build and run the container:

```sh
docker build -t ca-bench .
docker run --rm -it ca-bench
```

### Adding New Tasks or Models

Add new tasks to the tasks/ directory and update the corresponding config files in configs/.
Register new models/providers in provider/ and update model configs.

## Citation

If you use CA-Bench in your research, please cite:

```
@misc{ca-bench,
  author = {authors},
  title = {CA-Bench: Comprehensive Benchmark for LLMs and Multimodal Models},
  year = {2025},
  url = {https://github.com/phamtungthuy/ca-bench}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

OpenAI, HuggingFace, and other open-source contributors.
See individual files for additional attributions.
