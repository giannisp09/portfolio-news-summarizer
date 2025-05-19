#!/usr/bin/env python3
"""
Configuration file for the news summarization project.
Contains default settings and paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Default models
DEFAULT_BASELINE_MODEL = os.getenv("BASELINE_MODEL", "facebook/bart-large-cnn")
DEFAULT_FINETUNED_PATH = os.getenv("FINETUNED_PATH", str(MODELS_DIR / "finetuned"))

# API keys
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default parameters
DEFAULT_MAX_LENGTH = 150
DEFAULT_MIN_LENGTH = 40

# Training settings
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_NUM_EPOCHS = 3
DEFAULT_MAX_INPUT_LENGTH = 512
DEFAULT_MAX_TARGET_LENGTH = 128

# LoRA settings
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05 