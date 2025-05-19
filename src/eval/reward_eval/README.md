# Reward Model Evaluation

This directory contains scripts for evaluating text summaries using a reward model. The reward model takes a source text and its summary as input and produces a single quality score.

## Contents

- `reward_model_eval.py`: The main module containing the `RewardModelEvaluator` class and CLI functionality
- `reward_eval.py`: Example script demonstrating how to use the evaluator

## Requirements

- PyTorch
- Transformers
- Pandas
- NumPy
- tqdm
- datasets (for Hugging Face dataset support)

## How to Use

### Command-line Interface

The `reward_model_eval.py` script provides a command-line interface for evaluating summaries in a dataset:

#### Local Dataset Example

```bash
python reward_model_eval.py \
  --model_path ../../reward_models/coverage-reg-model \
  --dataset_path path/to/your/dataset.csv \
  --text_column text \
  --summary_column summary \
  --output_path reward_scores.json \
  --batch_size 16 \
  --max_length 512 \
  --device 0  # Use specific GPU
```

#### Hugging Face Dataset Example

```bash
python reward_model_eval.py \
  --model_path ../../reward_models/coverage-reg-model \
  --hf_dataset cnn_dailymail \
  --hf_subset 3.0.0 \
  --hf_split test \
  --text_column article \
  --summary_column highlights \
  --output_path reward_scores.json \
  --batch_size 16 \
  --device -1  # Auto-detect best device
```

### Arguments

#### Core Arguments
- `--model_path`: Path to the reward model (required)
- `--output_path`: Path to save evaluation results (default: "reward_model_scores.json")
- `--batch_size`: Batch size for evaluation (default: 16)
- `--max_length`: Maximum sequence length for tokenization (default: 512)
- `--device`: Device to use (-1 for auto-detect, 0+ for specific GPU, default: -1)

#### Dataset Loading Options (one required)
- `--dataset_path`: Path to a local dataset file (.csv, .json, .jsonl)
- `--hf_dataset`: Name of a Hugging Face dataset

#### Hugging Face Dataset Options
- `--hf_subset`: Subset/config of the Hugging Face dataset
- `--hf_split`: Split to use for the Hugging Face dataset (default: "test")

#### Column Specification
- `--text_column`: Column name for source texts (default: "text")
- `--summary_column`: Column name for summaries (default: "summary")

### Programmatic Usage

You can also use the `RewardModelEvaluator` class directly in your code:

```python
from reward_model_eval import RewardModelEvaluator

# Initialize the evaluator with device selection
evaluator = RewardModelEvaluator(
    model_path="path/to/model",
    device=0,  # Use specific GPU (or -1 for auto-detect, "cpu" for CPU, "cuda:1" for specific CUDA device)
    max_length=512
)

# Score a single text-summary pair
score = evaluator.score(
    text="Your source text here.",
    summary="Your summary here."
)
print(f"Reward score: {score:.4f}")

# Score a batch of text-summary pairs
texts = ["Text 1", "Text 2", "Text 3"]
summaries = ["Summary 1", "Summary 2", "Summary 3"]
scores = evaluator.batch_score(texts, summaries)

# Evaluate a dataset
import pandas as pd
df = pd.read_csv("your_data.csv")
results = evaluator.evaluate_dataset(
    dataset=df,
    text_column="text",
    summary_column="summary",
    batch_size=16
)
print(f"Average score: {results['mean_score']:.4f}")

# Load and evaluate a Hugging Face dataset
from reward_model_eval import load_huggingface_dataset
hf_dataset = load_huggingface_dataset(
    dataset_name="cnn_dailymail",
    subset="3.0.0",
    split="test",
    text_column="article",
    summary_column="highlights"
)
results = evaluator.evaluate_dataset(
    dataset=hf_dataset,
    text_column="article",
    summary_column="highlights",
    batch_size=16
)
```

## Example Usage

Run the example script to see the reward model in action:

```bash
python example_reward_eval.py
```

This will score example summaries using the reward model and display the results.

## Device Selection

The script supports automatic device detection with the following priority:
1. Explicitly specified GPU (when using `--device 0`, `--device 1`, etc.)
2. CUDA GPU (if available)
3. Apple Silicon MPS (Metal Performance Shaders, if available)
4. CPU (fallback)

## Reward Model

The reward model is a fine-tuned transformer model that evaluates the quality of a summary given its source text. It produces a single value where higher scores indicate better summaries. The model considers various aspects like:

- Factual correctness/consistency with the source
- Coverage of key information
- Conciseness and clarity
- Language quality

## Output Format

The evaluation results are saved as a JSON file with the following metrics:

```json
{
  "mean_score": 0.753,
  "std_score": 0.124,
  "min_score": 0.382,
  "max_score": 0.924,
  "median_score": 0.781,
  "scores": [0.782, 0.653, ...]
}
``` 