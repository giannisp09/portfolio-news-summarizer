# Portfolio News Summarizer

A tool for scraping, processing, and summarizing news articles related to stock tickers. This project demonstrates the use of natural language processing techniques to extract key information from financial news articles, enhancing the ability of investors to quickly digest market information.

## Features

- Scrapes news articles for specific stock tickers from Yahoo Finance and NewsAPI
- Cleans and preprocesses text data, removing boilerplate and normalizing content
- Generates summaries using pre-trained models (facebook/bart-large-cnn)
- Fine-tunes smaller, more efficient summarization models on financial news using PEFT/LoRA
- Evaluates summary quality using ROUGE, BERTScore, and factuality checks
- Provides a Streamlit UI for interacting with the application
- Portfolio-centric approach to managing tickers and news
- Automatic article categorization by topic (Earnings, Products, M&A, etc.)
- Keyword tracking across portfolio news
- Export capabilities for summaries and analysis (CSV, Excel, Markdown)
- Interactive dashboard with recent activity tracking
- Visual data exploration with charts and heatmaps

## Project Structure

```
portfolio-news-summarizer/
├── src/                    # Source code
│   ├── data_utils/        # Data processing and cleaning utilities
│   ├── scraping/          # News scraping modules
│   ├── finetune/          # Model fine-tuning code
│   ├── inference/         # Model inference and summarization
│   └── eval/             # Evaluation metrics and analysis
├── data/                  # Data storage
├── models/               # Model checkpoints and weights
├── results/             # Evaluation results and visualizations
├── ui/                  # Streamlit user interface
├── reward_models/       # Reward models for RLHF
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── Makefile          # Build automation
└── README.md         # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+
- pip or Docker
- (Optional) NewsAPI key - [Get one here](https://newsapi.org/register)
- (Optional) OpenAI API key - [Get one here](https://platform.openai.com/signup)

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-news-summarizer.git
cd portfolio-news-summarizer

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make setup

# Set up environment variables
cp env-template .env
# Edit .env with your API keys
```

### Using Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-news-summarizer.git
cd portfolio-news-summarizer

# Set up environment variables
cp env-template .env
# Edit .env with your API keys

# Build and run with Docker Compose
make docker
```

## Usage

### Basic Pipeline

The project provides a Makefile with common commands for running the pipeline:

```bash
# Scrape news for a specific ticker
make scrape TICKER=AAPL

# Run the complete pipeline for a ticker
make pipeline-sample TICKER=AAPL

# Start the Streamlit UI
make ui
```

### Advanced Usage

#### Data Collection

```bash
# Gather data for multiple tickers
make gather-data TICKERS="AAPL MSFT GOOGL TSLA"

# Scrape news for a specific ticker
make scrape TICKER=TSLA
```

#### Data Processing

```bash
# Clean and preprocess articles
make preprocess FILE=data/TSLA_articles_20230101_120000.json

# Generate baseline summaries
make baseline FILE=data/TSLA_articles_20230101_120000_cleaned.json
```

#### Model Training

```bash
# Prepare training data
make finetune-data

# Fine-tune the model
make finetune
```

#### Evaluation

```bash
# Run inference with both models
make inference FILE=data/TSLA_articles_20230101_120000_cleaned.json

# Evaluate summaries
make evaluate FILE=data/TSLA_articles_20230101_120000_comparison.json
```

## Development

### Setting Up Development Environment

1. Clone the repository
2. Create a virtual environment
3. Install development dependencies:
   ```bash
   make setup
   ```
4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_scraper.py
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run all checks:
```bash
make lint
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
- [Yahoo Finance](https://finance.yahoo.com/)

## Causal Language Models for Summarization

This project now supports using small instruction-tuned causal language models (LLMs) for news summarization. These are decoder-only models like Phi-2, TinyLlama, Gemma, etc., which are fine-tuned for following instructions.

### Available Small LLMs

The following small LLMs are supported:

- **TinyLlama** (1.1B parameters): A small but capable instruction-tuned LLaMA model
- **Phi-2** (2.7B parameters): Microsoft's small but powerful language model
- **Phi-3-mini** (3.8B parameters): Microsoft's newer instruction-tuned model
- **Gemma** (2B parameters): Google's lightweight instruction-tuned model
- **LLaMA-3-8B** (8B parameters): Meta's efficient instruction-tuned model

### Using Small LLMs for Baseline Summarization

To generate summaries using a small instruction-tuned LLM without fine-tuning:

```bash
python causal_baseline.py --input data/processed_articles.json --small-llm phi2
```

Other options:
- `--small-llm`: Choose from available models (phi2, phi3, tinyllama, gemma, llama3)
- `--model-name`: Use a custom model from Hugging Face
- `--device`: Specify GPU device (-1 for auto-detection)
- `--max-length`: Maximum length of the summary in tokens
- `--min-length`: Minimum length of the summary in tokens

### Fine-tuning Small LLMs

To fine-tune a small instruction-tuned LLM on your summarization dataset:

```bash
python causal_finetune.py --data-file data/train_data.json --small-llm phi2 --output-dir models/finetuned_phi2 --use-8bit
```

Options:
- `--data-file`: JSON file with training data
- `--hf-dataset`: Alternatively, use a dataset from Hugging Face
- `--small-llm`: Choose from available models (phi2, phi3, tinyllama, gemma, llama3)
- `--model`: Use a custom model from Hugging Face
- `--use-8bit`: Use 8-bit quantization for memory efficiency
- `--use-4bit`: Use 4-bit quantization (QLoRA) for even more memory efficiency
- `--lora-r`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA alpha (default: 16)
- `--lora-dropout`: LoRA dropout (default: 0.1)
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--epochs`: Number of training epochs
- `--use-wandb`: Log training metrics with Weights & Biases

### Inference with Fine-tuned LLMs

To generate summaries using your fine-tuned model:

```bash
python causal_inference.py --input data/test_articles.json --model-path models/finetuned_phi2/final --model-key phi2
```

Options:
- `--model-path`: Path to the fine-tuned model or adapter
- `--model-key`: Type of model for prompt formatting (phi2, phi3, etc.)
- `--use-8bit`: Use 8-bit quantization
- `--use-4bit`: Use 4-bit quantization
- `--temperature`: Sampling temperature (0.7 by default, 0 for greedy)
- `--num-beams`: Number of beams for beam search (1 by default)

### Memory Requirements

Small LLMs have different memory requirements:
- TinyLlama, Phi-2: Can run on 4GB+ GPU VRAM
- Phi-3-mini, Gemma: Require 8GB+ GPU VRAM
- LLaMA-3-8B: Requires 16GB+ GPU VRAM

Using 4-bit or 8-bit quantization can significantly reduce memory requirements. 