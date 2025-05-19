#!/usr/bin/env python3
"""
Script to fine-tune a small summarization model using PEFT/LoRA.
Designed to run on Colab or similar environments with limited GPU memory.
"""

import os
import json
import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

# Check for required packages and install if missing
required_packages = {
    #"torchtext": "0.17.1",
   # "bert-score": "0.3.13",
}

def check_install_package(package, version=None):
    """Check if package is installed, install if not."""
    try:
        __import__(package)
        print(f"Package {package} is already installed.")
    except ImportError:
        print(f"Package {package} not found. Installing...")
        if version:
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Package {package} has been installed.")

# Check and install required packages
for package, version in required_packages.items():
    check_install_package(package, version)

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    PeftConfig,
)

# Ensure models directory exists
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Default model
DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"

def get_best_device(device_param=-1):
    """
    Get the best available device for training.
    
    Args:
        device_param: User-specified device (-1 for auto-detect, 0+ for specific GPU/device)
        
    Returns:
        Device string for torch and HF
    """
    # If user specified a specific GPU device, try to use it
    if device_param >= 0 and torch.cuda.is_available() and device_param < torch.cuda.device_count():
        device_str = f"cuda:{device_param}"
        print(f"Using specified GPU device: {device_str} ({torch.cuda.get_device_name(device_param)})")
        return device_str
    
    # Auto-detection: Check for CUDA GPUs
    if torch.cuda.is_available():
        device_str = "cuda:0"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device_str
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_str = "mps"
        print("Using Apple Silicon MPS (Metal Performance Shaders)")
        return device_str
    
    # Fallback to CPU
    print("No GPU detected, using CPU for training (this will be slow)")
    return "cpu"

def load_data(data_prefix: str, max_samples: int = None) -> Dict[str, Dataset]:
    """
    Load data for fine-tuning.
    
    Args:
        data_prefix: Prefix for data files
        max_samples: Maximum number of samples to use per split
        
    Returns:
        Dictionary of datasets for training, validation, and test
    """
    datasets = {}
    
    for split in ["train", "val", "test"]:
        file_path = f"{data_prefix}_{split}.csv"
        df = pd.read_csv(file_path)
        
        if max_samples is not None and len(df) > max_samples and split == "train":
            df = df.sample(max_samples, random_state=42)
            
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        datasets[split] = dataset
        
        print(f"Loaded {len(dataset)} examples for {split}")
        
    return datasets


def load_data_from_json(json_file: str, split_ratio: Dict[str, float] = None, max_samples: int = None) -> Dict[str, Dataset]:
    """
    Load data for fine-tuning from a JSON file.
    
    Args:
        json_file: Path to JSON file containing the dataset
        split_ratio: Dictionary with split ratios for train, val, and test
                     (defaults to 0.8, 0.1, 0.1)
        max_samples: Maximum number of samples to use
        
    Returns:
        Dictionary of datasets for training, validation, and test
    """
    if split_ratio is None:
        split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    # Check that split ratios sum to 1
    total_ratio = sum(split_ratio.values())
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Split ratios sum to {total_ratio}, not 1.0")
        # Normalize ratios
        split_ratio = {k: v / total_ratio for k, v in split_ratio.items()}
    
    print(f"Loading data from JSON file: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        # Direct list of examples
        examples = data
    elif isinstance(data, dict) and "articles" in data:
        # Dictionary with "articles" key
        examples = data["articles"]
    elif isinstance(data, dict) and all(isinstance(data[k], list) for k in data):
        # Dictionary with split keys
        datasets = {}
        for split, items in data.items():
            df = pd.DataFrame(items)
            
            # Ensure required columns exist
            required_cols = {"text", "summary"}
            if not all(col in df.columns for col in required_cols):
                # Try to map columns if possible
                if "article" in df.columns and "text" not in df.columns:
                    df["text"] = df["article"]
                if "title" in df.columns and "summary" not in df.columns:
                    df["summary"] = df["title"]
                
                # Check for other possible column mappings
                text_candidates = ["text", "article", "content", "body", "cleaned_text"]
                summary_candidates = ["summary", "headline", "title", "labeled_summary", "reference_summary", "ground_truth"]
                
                for candidate in text_candidates:
                    if candidate in df.columns and "text" not in df.columns:
                        df["text"] = df[candidate]
                        break
                        
                for candidate in summary_candidates:
                    if candidate in df.columns and "summary" not in df.columns:
                        df["summary"] = df[candidate]
                        break
            
            # Ensure required columns exist after mapping
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"JSON data must contain 'text' and 'summary' fields for split {split}")
            
            if max_samples is not None and len(df) > max_samples and split == "train":
                df = df.sample(max_samples, random_state=42)
                
            dataset = Dataset.from_pandas(df)
            datasets[split] = dataset
            print(f"Loaded {len(dataset)} examples for {split}")
        
        return datasets
    else:
        raise ValueError("Unrecognized JSON format. Expected a list of examples or a dictionary with 'articles' key.")
    
    # Convert to DataFrame
    df = pd.DataFrame(examples)
    
    # Ensure required columns exist
    required_cols = {"text", "summary"}
    if not all(col in df.columns for col in required_cols):
        # Try to map columns if possible
        if "article" in df.columns and "text" not in df.columns:
            df["text"] = df["article"]
        if "title" in df.columns and "summary" not in df.columns:
            df["summary"] = df["title"]
        
        # Check for other possible column mappings
        text_candidates = ["text", "article", "content", "body", "cleaned_text"]
        summary_candidates = ["summary", "headline", "title", "labeled_summary", "reference_summary", "ground_truth"]
        
        for candidate in text_candidates:
            if candidate in df.columns and "text" not in df.columns:
                df["text"] = df[candidate]
                break
                
        for candidate in summary_candidates:
            if candidate in df.columns and "summary" not in df.columns:
                df["summary"] = df[candidate]
                break
    
    # Ensure required columns exist after mapping
    if not all(col in df.columns for col in required_cols):
        raise ValueError("JSON data must contain 'text' and 'summary' fields")
    
    # Apply max_samples
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data
    datasets = {}
    n = len(df)
    train_end = int(n * split_ratio["train"])
    val_end = train_end + int(n * split_ratio["val"])
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Convert to Hugging Face datasets and handle column names
    datasets["train"] = Dataset.from_pandas(train_df)
    datasets["val"] = Dataset.from_pandas(val_df)
    datasets["test"] = Dataset.from_pandas(test_df)
    
    print(f"Loaded {len(datasets['train'])} examples for train")
    print(f"Loaded {len(datasets['val'])} examples for val")
    print(f"Loaded {len(datasets['test'])} examples for test")
    
    return datasets


def load_data_from_huggingface(
    dataset_name: str,
    text_column: str = "text",
    summary_column: str = "summary", 
    max_samples: int = None,
    split_ratio: Dict[str, float] = None
) -> Dict[str, Dataset]:
    """
    Load a dataset from Hugging Face's Datasets Hub.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        text_column: Column name for the input text
        summary_column: Column name for the summary
        max_samples: Maximum number of samples to use (distributed proportionally across splits)
        split_ratio: Dictionary with split ratios (defaults to {"train": 0.8, "val": 0.1, "test": 0.1})
        
    Returns:
        Dictionary of datasets for training, validation, and test
    """
    from datasets import load_dataset
    
    # Set default split ratio if not provided
    if split_ratio is None:
        split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    # Ensure split ratios sum to 1
    total_ratio = sum(split_ratio.values())
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Split ratios sum to {total_ratio}, not 1.0. Normalizing.")
        split_ratio = {k: v / total_ratio for k, v in split_ratio.items()}
    
    print(f"Loading dataset {dataset_name} from Hugging Face")
    
    # Load the dataset
    try:
        hf_dataset = load_dataset(dataset_name)
    except Exception as e:
        raise ValueError(f"Error loading dataset {dataset_name}: {e}")
    
    # Identify available splits
    available_splits = list(hf_dataset.keys())
    print(f"Available splits: {available_splits}")
    
    # Map standard split names to available splits
    split_mapping = {
        "train": "train",
        "val": "validation" if "validation" in available_splits else "dev" if "dev" in available_splits else "val",
        "test": "test"
    }
    
    # Check if the dataset has the required splits
    datasets = {}
    original_sizes = {}
    
    for split_name, hf_split_name in split_mapping.items():
        if hf_split_name in available_splits:
            # Get the dataset for this split
            split_dataset = hf_dataset[hf_split_name]
            original_sizes[split_name] = len(split_dataset)
            
            # Verify column names exist or map to available columns
            columns = list(split_dataset.features.keys())
            
            # Map column names if needed
            mapped_text_column = text_column
            mapped_summary_column = summary_column
            
            # Common text column names
            text_candidates = ["text", "article", "document", "content", "input", "source", "body"]
            summary_candidates = ["summary", "highlights", "target", "output", "headline", "title", "abstract"]
            
            # If the exact column names aren't found, try common alternatives
            if text_column not in columns:
                for candidate in text_candidates:
                    if candidate in columns:
                        mapped_text_column = candidate
                        print(f"Mapped text column from '{text_column}' to '{mapped_text_column}'")
                        break
                else:
                    print(f"Warning: Could not find a suitable text column in {columns}")
                    
            if summary_column not in columns:
                for candidate in summary_candidates:
                    if candidate in columns:
                        mapped_summary_column = candidate
                        print(f"Mapped summary column from '{summary_column}' to '{mapped_summary_column}'")
                        break
                else:
                    print(f"Warning: Could not find a suitable summary column in {columns}")
            
            # Ensure mapped columns exist
            if mapped_text_column not in columns or mapped_summary_column not in columns:
                missing = []
                if mapped_text_column not in columns:
                    missing.append(mapped_text_column)
                if mapped_summary_column not in columns:
                    missing.append(mapped_summary_column)
                raise ValueError(f"Required columns {missing} not found in dataset. Available columns: {columns}")
            
            datasets[split_name] = split_dataset
            print(f"Loaded {len(split_dataset)} examples for {split_name}")
        else:
            print(f"Warning: Split '{split_name}' not found in dataset. Available splits: {available_splits}")
    
    # Check if we have at least a training set
    if "train" not in datasets:
        raise ValueError(f"Training split not found in dataset {dataset_name}. Available splits: {available_splits}")
    
    # Create validation/test splits if missing by splitting the training set
    if "val" not in datasets and "train" in datasets:
        print("Creating validation split from training data")
        train_val = datasets["train"].train_test_split(test_size=split_ratio["val"] / (split_ratio["train"] + split_ratio["val"]), seed=42)
        datasets["train"] = train_val["train"]
        datasets["val"] = train_val["test"]
        original_sizes["val"] = len(datasets["val"])
        
    if "test" not in datasets and "train" in datasets:
        print("Creating test split from training data")
        if "val" in datasets:
            # Already have a validation set, so just split training
            remaining_ratio = split_ratio["train"] + split_ratio["test"]
            test_portion = split_ratio["test"] / remaining_ratio
            train_test = datasets["train"].train_test_split(test_size=test_portion, seed=43)
            datasets["train"] = train_test["train"]
            datasets["test"] = train_test["test"]
        else:
            # No validation set either, split into 3 parts
            train_val_test = datasets["train"].train_test_split(test_size=split_ratio["val"] + split_ratio["test"], seed=42)
            val_test_ratio = split_ratio["val"] / (split_ratio["val"] + split_ratio["test"])
            val_test = train_val_test["test"].train_test_split(test_size=1-val_test_ratio, seed=43)
            datasets["train"] = train_val_test["train"]
            datasets["val"] = val_test["train"]
            datasets["test"] = val_test["test"]
        
        original_sizes["test"] = len(datasets["test"])
    
    # Apply max_samples to each split according to split ratio
    if max_samples is not None:
        print(f"Applying max_samples={max_samples} proportionally across splits")
        
        # Calculate target sizes for each split
        target_sizes = {
            split: min(int(max_samples * split_ratio[split]), original_sizes[split])
            for split in datasets.keys()
        }
        
        # Resize each split
        for split_name, split_dataset in datasets.items():
            target_size = target_sizes[split_name]
            if len(split_dataset) > target_size:
                print(f"Resizing {split_name} split from {len(split_dataset)} to {target_size} examples")
                datasets[split_name] = split_dataset.select(range(target_size))
    
    # Rename columns to expected format in all splits
    for split_name, split_dataset in datasets.items():
        # Map column names if needed
        columns = list(split_dataset.features.keys())
        
        # First try to find the best matches for the columns
        if text_column not in columns:
            for candidate in text_candidates:
                if candidate in columns:
                    mapped_text_column = candidate
                    break
            else:
                mapped_text_column = text_column  # Use original as fallback
                
        if summary_column not in columns:
            for candidate in summary_candidates:
                if candidate in columns:
                    mapped_summary_column = candidate
                    break
            else:
                mapped_summary_column = summary_column  # Use original as fallback
        
        # Now rename columns if needed and if they exist
        if mapped_text_column in columns and mapped_text_column != "text":
            datasets[split_name] = split_dataset.rename_column(mapped_text_column, "text")
        if mapped_summary_column in columns and mapped_summary_column != "summary":
            datasets[split_name] = split_dataset.rename_column(mapped_summary_column, "summary")
        
    # Verify final sizes
    for split_name, split_dataset in datasets.items():
        print(f"Final {split_name} split: {len(split_dataset)} examples")
    
    return datasets


def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    """
    Tokenize inputs and targets.
    """
    # Add debug output to check what examples look like
    print(f"Preprocessing batch of size {len(examples['text']) if 'text' in examples else 'unknown'}")
    
    # Defensive handling of inputs
    if "text" not in examples or "summary" not in examples:
        print(f"WARNING: Missing required columns. Available columns: {list(examples.keys())}")
        # Try to map columns if standard ones not found
        if "cleaned_text" in examples and "text" not in examples:
            examples["text"] = examples["cleaned_text"]
        elif "article" in examples and "text" not in examples:
            examples["text"] = examples["article"]
        elif "document" in examples and "text" not in examples:
            examples["text"] = examples["document"]
            
        if "title" in examples and "summary" not in examples:
            examples["summary"] = examples["title"]
            
        # Check again after mapping
        if "text" not in examples or "summary" not in examples:
            print("ERROR: Could not find or map required columns 'text' and 'summary'")
            # Create empty dummy values to avoid errors
            batch_size = len(next(iter(examples.values())))
            examples["text"] = [""] * batch_size
            examples["summary"] = [""] * batch_size
    
    inputs = examples["text"]
    targets = examples["summary"]
    
    # Ensure inputs and targets are strings
    inputs = [str(text) if text is not None else "" for text in inputs]
    targets = [str(summary) if summary is not None else "" for summary in targets]
    
    # Create model inputs with error handling
    try:
        model_inputs = tokenizer(
            inputs, max_length=max_input_length, padding="max_length", truncation=True
        )

        # Set up the targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding="max_length", truncation=True
            )
            
        model_inputs["labels"] = labels["input_ids"]
    except Exception as e:
        print(f"Error during tokenization: {str(e)}")
        # Create dummy tokenized inputs to prevent crashes
        batch_size = len(inputs)
        model_inputs = {
            "input_ids": [[tokenizer.pad_token_id] * max_input_length] * batch_size,
            "attention_mask": [[0] * max_input_length] * batch_size,
            "labels": [[tokenizer.pad_token_id] * max_target_length] * batch_size,
        }
    
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """
    Compute evaluation metrics: ROUGE, BLEU, and BERTScore.
    """
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    import nltk
    from torchtext.data.metrics import bleu_score as torchtext_bleu_score
    
    # Make sure NLTK has necessary data for tokenization
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # With HF's Trainer, eval_pred is an EvalPrediction object, not a tuple
    if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        # Fallback for tuple format
        predictions, labels = eval_pred
    
    # Error handling for empty predictions or labels
    if predictions is None or labels is None or len(predictions) == 0 or len(labels) == 0:
        print(f"WARNING: Empty predictions or labels. predictions: {type(predictions)}, labels: {type(labels)}")
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'bleu': 0.0,
            'bertscore': 0.0,
        }
    
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
        
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(label, pred)
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure
        
        # Average ROUGE scores
        count = len(decoded_preds)
        rouge1 = rouge1 / count if count > 0 else 0
        rouge2 = rouge2 / count if count > 0 else 0
        rougeL = rougeL / count if count > 0 else 0
        
        # BLEU score using torchtext
        tokenized_preds = [nltk.word_tokenize(pred.lower()) for pred in decoded_preds]
        tokenized_labels = [nltk.word_tokenize(label.lower()) for label in decoded_labels]
        
        # Filter out empty tokenizations
        valid_examples = [(pred, label) for pred, label in zip(tokenized_preds, tokenized_labels) if pred and label]
        if valid_examples:
            valid_preds, valid_labels = zip(*valid_examples)
            # Convert to format expected by torchtext (list of candidate token sequences, list of lists of reference token sequences)
            bleu = torchtext_bleu_score(list(valid_preds), [[ref] for ref in valid_labels])
        else:
            bleu = 0.0
        
        # BERTScore - compute on a sample if there are many predictions
        try:
            if len(decoded_preds) > 50:
                # Sample 50 examples for BERTScore to keep evaluation time reasonable
                indices = np.random.choice(len(decoded_preds), 50, replace=False)
                sample_preds = [decoded_preds[i] for i in indices]
                sample_labels = [decoded_labels[i] for i in indices]
                
                # Check for empty predictions or references
                if len(sample_preds) == 0 or len(sample_labels) == 0:
                    bertscore = 0.0
                else:
                    _, _, F1 = bert_score(sample_preds, sample_labels, lang="en", verbose=False)
                    bertscore = F1.mean().item()
            else:
                # Check for empty predictions or references
                if len(decoded_preds) == 0 or len(decoded_labels) == 0:
                    bertscore = 0.0
                else:
                    _, _, F1 = bert_score(decoded_preds, decoded_labels, lang="en", verbose=False)
                    bertscore = F1.mean().item()
        except Exception as e:
            print(f"Error computing BERTScore: {str(e)}")
            bertscore = 0.0
    
        return {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'bleu': bleu,
            'bertscore': bertscore,
        }
    except Exception as e:
        print(f"Error in compute_metrics: {str(e)}")
        # Return default metrics on error
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'bleu': 0.0,
            'bertscore': 0.0,
        }


def setup_trainer(
    model,
    tokenizer,
    datasets,
    output_dir,
    max_input_length=512,
    max_target_length=128,
    batch_size=8,
    lr=5e-5,
    weight_decay=0.01,
    num_epochs=3,
    use_wandb=False,
    project_name="news-summarizer",
    device="cpu",
):
    """
    Set up the trainer for fine-tuning.
    
    Args:
        model: Model to fine-tune
        tokenizer: Tokenizer
        datasets: Dictionary of datasets
        output_dir: Output directory
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        num_epochs: Number of epochs
        use_wandb: Whether to use wandb for tracking
        project_name: wandb project name
        device: Device to use for training
        
    Returns:
        Hugging Face Trainer
    """
    # Preprocess data
    def preprocess(examples):
        return preprocess_function(
            examples, tokenizer, max_input_length, max_target_length
        )
    
    tokenized_datasets = {
        split: dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names
        )
        for split, dataset in datasets.items()
    }
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt",
    )
    
    # Setup metrics
    metric_function = lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    
    # Check if we should use fp16 (mixed precision training)
    # FP16 is only supported on CUDA devices, not MPS or CPU
    use_fp16 = device.startswith("cuda")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bertscore",
        fp16=use_fp16,
        # Add wandb reporting
        report_to="wandb" if use_wandb else "none",
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metric_function,
    )
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small summarization model using PEFT/LoRA")
    
    # Data loading options (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-prefix", type=str,
                         help="Prefix for CSV data files (will load <prefix>_train.csv, <prefix>_val.csv, etc.)")
    data_group.add_argument("--json-file", type=str,
                         help="Path to JSON file containing the dataset")
    data_group.add_argument("--dataset-name", type=str,
                         help="Name of the dataset on Hugging Face")
    
    # Split ratio for JSON data
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Proportion of data to use for training when loading from JSON")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Proportion of data to use for validation when loading from JSON")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Proportion of data to use for testing when loading from JSON")
    
    # Column names for Hugging Face datasets
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column name for input text in Hugging Face dataset")
    parser.add_argument("--summary-column", type=str, default="summary",
                        help="Column name for summary in Hugging Face dataset")
    
    # Model and training parameters
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "finetuned"),
                        help="Output directory")
    parser.add_argument("--max-input-length", type=int, default=512,
                        help="Maximum input sequence length")
    parser.add_argument("--max-target-length", type=int, default=128,
                        help="Maximum target sequence length")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to use")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--device", type=int, default=-1,
                        help="Device to use (-1 for auto-detection, 0+ for specific GPU)")
                        
    # Wandb parameters
    parser.add_argument("--use-wandb", action="store_true", 
                        help="Whether to use Weights & Biases for tracking")
    parser.add_argument("--wandb-project", type=str, default="news-summarizer",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases entity (team) name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Weights & Biases run name")
                        
    args = parser.parse_args()
    
    # Get the best available device
    device = get_best_device(args.device)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize wandb if requested
    if args.use_wandb:
        # Create a unique run name if not provided
        run_name = args.wandb_name or f"finetune-{args.model.split('/')[-1]}-{Path(args.output_dir).name}"
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "model": args.model,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_epochs": args.num_epochs,
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "device": device,
            }
        )
    
    # Create split ratio dictionary
    split_ratio = {
        "train": args.train_split,
        "val": args.val_split,
        "test": args.test_split
    }
    
    # Load datasets
    if args.data_prefix:
        datasets = load_data(args.data_prefix, args.max_samples)
    elif args.json_file:
        # Load from JSON file
        datasets = load_data_from_json(args.json_file, split_ratio, args.max_samples)
    elif args.dataset_name:
        datasets = load_data_from_huggingface(args.dataset_name, args.text_column, args.summary_column, args.max_samples, split_ratio)
    
    # Log dataset sizes
    if args.use_wandb:
        wandb.log({
            "train_examples": len(datasets["train"]),
            "val_examples": len(datasets["val"]),
            "test_examples": len(datasets["test"]),
        })
    
    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    
    # Move model to appropriate device
    model = model.to(device)
    
    # Configure LoRA
    print("Configuring LoRA")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    )
    
    # Wrap model with LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Setup trainer
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        output_dir=str(output_dir),
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project,
        device=device,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model(str(output_dir / "final"))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    if "test" in datasets and datasets["test"] is not None and len(datasets["test"]) > 0:
        try:
            # Ensure test dataset has the right columns
            test_dataset = datasets["test"]
            
            # Make sure the test dataset has required columns
            if "text" not in test_dataset.column_names or "summary" not in test_dataset.column_names:
                # Try to map columns
                if "cleaned_text" in test_dataset.column_names and "text" not in test_dataset.column_names:
                    test_dataset = test_dataset.rename_column("cleaned_text", "text")
                if "document" in test_dataset.column_names and "text" not in test_dataset.column_names:
                    test_dataset = test_dataset.rename_column("document", "text")
                if "title" in test_dataset.column_names and "summary" not in test_dataset.column_names:
                    test_dataset = test_dataset.rename_column("title", "summary")
            
            print(f"Test dataset columns: {test_dataset.column_names}")
            
            # Use a smaller test set if needed for memory constraints
            if len(test_dataset) > 20:
                print(f"Using a subset of {min(20, len(test_dataset))} examples for evaluation")
                test_dataset = test_dataset.select(range(min(20, len(test_dataset))))
            
            # Preprocess test dataset
            def preprocess_test(examples):
                inputs = examples["text"]
                targets = examples["summary"]
                
                # Ensure inputs and targets are strings
                inputs = [str(text) if text is not None else "" for text in inputs]
                targets = [str(summary) if summary is not None else "" for summary in targets]
                
                model_inputs = tokenizer(
                    inputs, max_length=args.max_input_length, padding="max_length", truncation=True
                )
                
                # Set up the targets
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        targets, max_length=args.max_target_length, padding="max_length", truncation=True
                    )
                    
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            
            tokenized_test = test_dataset.map(
                preprocess_test,
                batched=True,
                remove_columns=test_dataset.column_names
            )
            
            # Use the trainer to evaluate
            results = trainer.evaluate(tokenized_test)
            
            # Log final test results to wandb
            if args.use_wandb:
                wandb.log({f"test_{k}": v for k, v in results.items()})
            
            # Save results
            with open(output_dir / "test_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Evaluation complete. Results: {results}")
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping full evaluation due to error.")
            
            # If standard evaluation fails, try to at least generate a sample prediction
            try:
                print("Generating a sample prediction...")
                model.eval()
                with torch.no_grad():
                    # Get a sample text
                    sample_text = str(test_dataset[0]["text"])
                    sample_inputs = tokenizer(
                        sample_text, 
                        max_length=args.max_input_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Generate a prediction
                    generated_ids = model.generate(
                        input_ids=sample_inputs["input_ids"],
                        attention_mask=sample_inputs["attention_mask"],
                        max_length=args.max_target_length,
                    )
                    
                    # Decode the prediction
                    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    print(f"Generated summary: {prediction}")
                    
                    # Save the sample prediction
                    with open(output_dir / "sample_prediction.txt", "w") as f:
                        f.write(f"Input text: {sample_text[:500]}...\n\nGenerated summary: {prediction}")
                    
                    print("Sample prediction saved to sample_prediction.txt")
            except Exception as sample_error:
                print(f"Error generating sample prediction: {str(sample_error)}")
    else:
        print("Warning: Test dataset is empty or not available. Skipping evaluation.")
    
    print(f"Fine-tuning complete. Model saved to {output_dir}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 