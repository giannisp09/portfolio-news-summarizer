#!/bin/bash

# Create necessary directories
mkdir -p reward_models/coverage-reg-model
mkdir -p models/finetuned

# Download reward model
echo "Downloading reward model..."
wget -O reward_models/coverage-reg-model/model.safetensors https://huggingface.co/your-username/portfolio-news-summarizer/resolve/main/reward_models/coverage-reg-model/model.safetensors

# Download fine-tuned models (if available)
echo "Downloading fine-tuned models..."
wget -O models/finetuned/bart-large-cnn https://huggingface.co/your-username/portfolio-news-summarizer/resolve/main/models/finetuned/bart-large-cnn

echo "Download complete!" 