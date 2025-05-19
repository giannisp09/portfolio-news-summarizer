.PHONY: setup clean data scrape preprocess baseline finetune-data finetune inference evaluate ui docker docker-down help gather-data lint test format

# Default ticker to use if none specified
TICKER ?= AAPL

# Check if file exists for data processing
define check_file
	@if [ ! -f $(1) ]; then \
		echo "Error: File $(1) not found"; \
		exit 1; \
	fi
endef

help:
	@echo "Portfolio News Summarizer Makefile"
	@echo "Usage:"
	@echo "  make setup         - Install dependencies and download required models"
	@echo "  make clean         - Remove cached files and temporary directories"
	@echo "  make scrape        - Scrape news articles (use TICKER=SYMBOL to specify ticker)"
	@echo "  make preprocess    - Clean and preprocess scraped articles (pass FILE=path/to/file.json)"
	@echo "  make baseline      - Generate baseline summaries (pass FILE=path/to/file.json)"
	@echo "  make finetune-data - Prepare data for fine-tuning"
	@echo "  make finetune      - Fine-tune the summarization model"
	@echo "  make inference     - Run inference with both models (pass FILE=path/to/file.json)"
	@echo "  make evaluate      - Evaluate summaries (pass FILE=path/to/file.json)"
	@echo "  make ui            - Run the Streamlit UI"
	@echo "  make docker        - Build and run the Docker container"
	@echo "  make docker-down   - Stop the Docker container"
	@echo "  make gather-data   - Gather a large dataset from multiple tickers"
	@echo "  make lint          - Run code quality checks"
	@echo "  make test          - Run tests"
	@echo "  make format        - Format code using black and isort"
	@echo ""
	@echo "Example usage:"
	@echo "  make scrape TICKER=TSLA"
	@echo "  make preprocess FILE=data/TSLA_articles_20230101_120000.json"
	@echo "  make gather-data TICKERS=\"AAPL MSFT GOOGL TSLA\""

setup:
	@echo "Upgrading pip..."
	python -m pip install --upgrade pip
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Installing lxml with html_clean extras (required for newspaper3k)..."
	pip install "lxml[html_clean]"
	@echo "Downloading spaCy model..."
	python -m spacy download en_core_web_sm
	@echo "Creating necessary directories..."
	mkdir -p data models/finetuned results reward_models
	@echo "Setup complete."

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@echo "Clean complete."

gather-data:
	@echo "Gathering data from multiple tickers..."
ifdef TICKERS
	python src/scraping/gather_training_data.py --tickers $(TICKERS) --articles-per-ticker 30
else
	python src/scraping/gather_training_data.py --articles-per-ticker 30
endif
	@echo "Data gathering complete."

scrape:
	@echo "Scraping news articles for $(TICKER)..."
	python src/scraping/scraper.py --ticker $(TICKER)
	@echo "Scraping complete."

preprocess:
	@$(call check_file,$(FILE))
	@echo "Preprocessing articles from $(FILE)..."
	python src/data_utils/cleaner.py --input $(FILE)
	@echo "Preprocessing complete."

baseline:
	@$(call check_file,$(FILE))
	@echo "Generating baseline summaries for $(FILE)..."
	python src/inference/baseline.py --input $(FILE)
	@echo "Baseline summarization complete."

finetune-data:
	@echo "Preparing data for fine-tuning..."
	python src/finetune/prepare_training_data.py --source cnn_dailymail --subset-size 1000
	@echo "Data preparation complete."

finetune:
	@echo "Fine-tuning the summarization model..."
	python src/finetune/finetune.py --data-prefix data/finetune
	@echo "Fine-tuning complete."

inference:
	@$(call check_file,$(FILE))
	@echo "Running inference with both models on $(FILE)..."
	python src/inference/inference.py --input $(FILE)
	@echo "Inference complete."

evaluate:
	@$(call check_file,$(FILE))
	@echo "Evaluating summaries from $(FILE)..."
	python src/eval/metrics.py --input $(FILE)
	@echo "Evaluation complete."

ui:
	@echo "Starting Streamlit UI..."
	streamlit run ui/app.py

docker:
	@echo "Building and running Docker container..."
	docker-compose up --build

docker-down:
	@echo "Stopping Docker container..."
	docker-compose down

lint:
	@echo "Running code quality checks..."
	black --check .
	isort --check-only .
	flake8 .
	mypy src tests
	@echo "Lint complete."

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "Tests complete."

format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "Formatting complete."

# Pipeline targets to run multiple steps
pipeline-sample: scrape
	@echo "Running complete pipeline for $(TICKER)..."
	$(eval LATEST_FILE := $(shell ls -t data/$(TICKER)_articles_*.json | head -1))
	@echo "Using latest file: $(LATEST_FILE)"
	make preprocess FILE=$(LATEST_FILE)
	$(eval CLEANED_FILE := $(shell ls -t data/$(TICKER)_articles_*_cleaned.json | head -1))
	make baseline FILE=$(CLEANED_FILE)
	$(eval BASELINE_FILE := $(shell ls -t data/$(TICKER)_articles_*_baseline_summaries.json | head -1))
	make inference FILE=$(CLEANED_FILE)
	$(eval COMPARISON_FILE := $(shell ls -t data/$(TICKER)_articles_*_comparison.json | head -1))
	make evaluate FILE=$(COMPARISON_FILE)
	@echo "Pipeline complete!" 