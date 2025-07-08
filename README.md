# CA-Bench

# The Challenge

Conventional AI research has largely centered on developing monolithic models optimized for isolated tasks, often neglecting the necessity of composability for solving complex, real-world problems. The significant advancements in large language models (LLMs) have ignited interest in their capacity to tackle intricate tasks across diverse domains. Despite these impressive capabilities, the cognitive faculties of state-of-the-art LLMs often lack depth and robustness when confronted with tasks demanding multi-step reasoning and the integration of multimodal data.

To address this challenge, we introduce CA-Bench, a benchmark engineered to assess the proficiency of LLM-based agents in solving complex tasks through the compositional reuse of pretrained models. Our objective is to furnish a standardized benchmark that catalyzes the development of more resilient and generalizable composable AI systems.

# Key Features

- **Task Suite**: Includes a diverse set of tasks at node, chain, and graph levels.
- **Automated Evaluation**: Supports multiple evaluation metrics (accuracy, BLEU, CodeBLEU, F1, R2, semantic similarity, etc.).
- **Workflow Generation**: Automatically generates and runs workflows for each task.
- **Extensible**: Easily add new tasks, models, and evaluation metrics.
- **Cost Tracking**: Tracks API usage and cost for supported providers (e.g., OpenAI).
- **Multimodal Support**: Handles text, image, audio, and video tasks.

# The Benchmark Dataset

The construction of a dataset for composable AI problems, complete with corresponding ground-truth solutions and tests, is a substantial undertaking. Existing benchmarks often concentrate on evaluating the intermediate planning process, yet they lack executable ground-truth pipelines or expected outputs required for robust end-to-end evaluation.

Motivated by these limitations, we present CA-Bench: a dataset of 70 composable AI problems. Each problem is annotated with a corresponding solution pipeline, which includes invocations of both logic modules and pretrained models. The benchmark is designed to support the systematic and reproducible evaluation of composable AI systems across a wide spectrum of task complexities.

# Evaluation Spectrum: Baseline and Upper Bound

To provide a comprehensive performance spectrum, CA-Bench defines both a lower and an upper performance bound for each task.

## Zero-Shot Baseline (Lower Bound)

To establish a foundational performance level, we define a Zero-Shot Baseline. This approach involves prompting a general-purpose LLM (e.g., GPT-4o-mini) to solve a given task directly, without any explicit guidance on tool composition or access to the toolset. This baseline represents the unassisted reasoning capability of the LLM and serves as the lower bound for performance. Any effective agent must significantly outperform this baseline.

## Human-Designed Solution (Upper Bound)

To define the performance ceiling, we introduce a Human-Designed Upper Bound (or Oracle). For each task, we provide a manually crafted, optimal solution pipeline that represents the best possible sequence of tool invocations to achieve the desired outcome. This "oracle" solution is not meant to be a competitor but rather a performance target. It allows us to measure how close an automated agent's solution is to the theoretical maximum achievable within the provided framework, providing a clear metric for optimality.

# Getting Started

## Directory Structure

```text
├── configs/ # Configuration files for models and tasks
├── data/ # Model metadata
│   └── model_desc.jsonl # List of ML models
├── provider/ # LLM provider integrations (OpenAI, etc.)
├── scripts/ # Main scripts for evaluation, data download, etc.
├── server/ # API server for model inference
├── templates/ # Prompt and workflow templates
├── utils/ # Utility modules (token counting, cost management, etc.)
├── Dockerfile # Docker support
├── pyproject.toml # Python dependencies for evaluation
├── requirements.txt # Python dependencies for serving model
└── README.md # This file
```

## Prerequisites

- Model Hosting: ML models can be accessed from Hugging Face (if publicly available) or hosted locally via the provided model server.
- Serving Environment: To run the local model server, see dependencies in requirements.txt. We highly recommend using Docker for this.
- Evaluation Environment: To run the evaluation scripts, see dependencies in pyproject.toml.

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-org/ca-bench.git
   cd ca-bench
   ```

2. Setup Serving Models:
   Setup Option A: Docker (Recommended)

- Build and run the model serving container in the background:
  ```sh
  docker compose up -d
  ```

Setup Option B: Local Environment

- Install dependencies for serving models (if not using Docker):
  ```sh
  pip install -r requirements.txt
  ```

3. Install dependencies for the evaluation framework:

   ```sh
   poetry install
   ```

4. Download benchmark datasets:
   ```
   python -m scripts.download_data --datasets [tasks, human_design, zeroshot, all]
   ```

- Download options:
  - tasks : download all 70 tasks (include node, chain and graph level)
  - human_design : download all human-design solution for all tasks
  - zeroshot : download all zeroshot solution for all tasks
  - all : download all tasks, human_design, zeroshot.

## Running Evaluations

To evaluate a solution (e.g., zero-shot, human-designed, or your own solution) on a specific task:

```sh
python -m scripts.evaluate.py
--range [task, node, chain, graph, all] (*required)
--result_folder_name <TEXT> (*required)
--folder_name <task_folder>
--evaluation_type [calculate_score, generate_and_run_workflow, run_workflow, generate_workflow, all] (*required)
```

- The **--range** argument specifies the scope of execution or analysis. This is the required parameter and must be set to one of the following values:
  - task: Operates on a single task only, require **--folder_name** parameter for name of task.
  - node: Operates on all tasks in node-level or on a single taks in node-level with **--folder_name** parameter
  - chain: Operates on all tasks in chain-level or on a single taks in chain-level with **--folder_name** parameter
  - graph: Operates on all tasks in graph-level or on a single taks in graph-level with **--folder_name** parameter
  - all: Operates on all 70 tasks or on a single taks with **--folder_name** parameter

For all available options, run:

```sh
python evaluate.py --help
```

## Contributing and Extensibility

We welcome contributions to CA-Bench. To extend the benchmark:

### Add New ML Model:

The agent's capabilities are defined by the set of available models. To register a new model and make it available as a tool for the agent, you need to add its metadata to the data/model_desc.jsonl file.

This file uses the JSON Lines format, where each line is a separate JSON object describing one model. Each object must contain the following keys:

- id: The model's identifier on Hugging Face (e.g., "google/vit-base-patch16-224").
- tag: A functional category tag (e.g., "image-classification", "text-generation"). This helps in filtering and selecting appropriate tools.
- desc: A concise, natural language description of the model's function. This description is provided to the LLM agent to help it understand what the tool does and when to use it.

Example:
To add an image classification model and a text-to-image model, you would add the following lines to data/model_desc.jsonl:

{"id": "google/vit-base-patch16-224", "tag": "image-classification", "desc": "A model that classifies the main subject of an image. Input is an image, output is a text label (e.g., 'cat', 'airplane')."}

Providing a clear and accurate description in the desc field is crucial for the agent's planning and reasoning performance.

### Add New Tasks:

Place new task definitions in the **tasks/** directory and update the corresponding configuration files in configs/.

### Add New LLMs:

Register new LLMs or providers in the **provider/** directory and update the LLMs configurations.

Please refer to our contribution guidelines for more details.

# Cite Us

If you use CA-Bench in your research, please cite:

```
@misc{ca-bench,
  title = {CA-Bench: A comprehensive benchmark for Composable AI systems},
  author = {Pham, Tung-Thuy and Luong, Duy-Quan and Duong, Minh-Quan},
  year = {2025},
  url = {https://github.com/phamtungthuy/ca-bench}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

We extend our gratitude to OpenAI, HuggingFace, and the broader open-source community for their invaluable contributions. See individual files for additional attributions.
