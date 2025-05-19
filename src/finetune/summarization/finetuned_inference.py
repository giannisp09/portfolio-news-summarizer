#!/usr/bin/env python3
"""
Script to generate summaries using the fine-tuned model.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time

import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
from peft import PeftModel, PeftConfig

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Define default model directories
DEFAULT_MODEL_DIR = Path("models/finetuned/final")
MODELS_DIR = Path("models")


def get_best_device(device_param=-1):
    """
    Get the best available device for inference.
    
    Args:
        device_param: User-specified device (-1 for auto-detect, 0+ for specific GPU/device)
        
    Returns:
        Device string and numeric device id
    """
    # If user specified a specific GPU device, try to use it
    if device_param >= 0 and torch.cuda.is_available() and device_param < torch.cuda.device_count():
        device_str = f"cuda:{device_param}"
        print(f"Using specified GPU device: {device_str} ({torch.cuda.get_device_name(device_param)})")
        return device_str, device_param
    
    # Auto-detection: Check for CUDA GPUs
    if torch.cuda.is_available():
        device_str = "cuda:0"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device_str, 0
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_str = "mps"
        print("Using Apple Silicon MPS (Metal Performance Shaders)")
        return device_str, "mps"
    
    # Fallback to CPU
    print("No GPU detected, using CPU for inference (this will be slower)")
    return "cpu", -1


def load_articles(input_file: str) -> List[Dict[str, Any]]:
    """
    Load processed articles from a JSON file.
    
    Args:
        input_file: Path to the JSON file containing processed articles
        
    Returns:
        List of article dictionaries
    """
    with open(input_file, 'r') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {input_file}")
    return articles


def load_huggingface_dataset(
    dataset_name: str,
    split: str = "test",
    subset: Optional[str] = None,
    text_column: str = "text",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load a dataset from Hugging Face datasets hub.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Dataset split to use (default: "test")
        subset: Dataset configuration/subset
        text_column: Name of the column containing the text to summarize
        limit: Maximum number of examples to load (None for all)
        
    Returns:
        List of article dictionaries
    """
    print(f"Loading Hugging Face dataset: {dataset_name}" + (f", subset: {subset}" if subset else ""))
    
    # Load the dataset
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    # Verify the text column exists
    if text_column not in dataset.column_names:
        raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {dataset.column_names}")
    
    # Convert to list of dictionaries
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    articles = []
    for item in dataset:
        article = {
            "text": item[text_column],
            "id": item.get("id", str(len(articles))),
            "title": item.get("title", "")
        }
        articles.append(article)
    
    print(f"Loaded {len(articles)} examples from Hugging Face dataset")
    return articles


def setup_summarization_pipeline(model_dir: str, device: int = -1) -> pipeline:
    """
    Set up the Hugging Face summarization pipeline with a fine-tuned model.
    
    Args:
        model_dir: Directory containing the fine-tuned model
        device: Device to use (-1 for auto-detect, 0+ for specific GPU)
        
    Returns:
        Hugging Face pipeline for summarization
    """
    print(f"Loading fine-tuned model from: {model_dir}")
    
    # Determine device
    device_str, device_id = get_best_device(device)
    
    # Check if model directory exists
    model_path = Path(model_dir)
    if not model_path.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    
    # Load config to get the base model
    try:
        # First, try to load as a PEFT model
        peft_config_path = model_path / "adapter_config.json"
        if peft_config_path.exists():
            # It's a PEFT/LoRA model
            print("Loading as PEFT/LoRA model")
            config = PeftConfig.from_pretrained(model_path)
            base_model_name = config.base_model_name_or_path
            
            # Load base model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            
            # Load the PEFT adapter
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # It's a regular model (full fine-tuned or base)
            print("Loading as regular fine-tuned model")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Falling back to default model: facebook/bart-large-cnn")
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    
    # Move model to the appropriate device
    model = model.to(device_str)
    
    # Create pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device_id
    )
    
    return summarizer


def generate_summaries(
    articles: List[Dict[str, Any]],
    summarizer,
    max_length: int = 150,
    min_length: int = 40
) -> List[Dict[str, Any]]:
    """
    Generate summaries for each article using the fine-tuned model.
    
    Args:
        articles: List of article dictionaries
        summarizer: Hugging Face summarization pipeline
        max_length: Maximum length of the summary in tokens
        min_length: Minimum length of the summary in tokens
        
    Returns:
        List of article dictionaries with summaries added
    """
    articles_with_summaries = []
    
    for article in tqdm(articles, desc="Generating summaries with fine-tuned model"):
        # Get the text to summarize (prefer cleaned_text if available)
        text = article.get("cleaned_text", article.get("text", ""))
        
        # Skip articles without text
        if not text:
            continue
            
        # Generate summary
        try:
            summary = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            # Add summary to article
            article_with_summary = article.copy()
            article_with_summary["finetuned_summary"] = summary[0]["summary_text"]
            
            # Keep baseline summary if already present
            if "baseline_summary" in article:
                article_with_summary["baseline_summary"] = article["baseline_summary"]
                
            articles_with_summaries.append(article_with_summary)
            
            # Slight delay to avoid overloading GPU
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error generating summary for article '{article.get('title', 'Unknown')}': {str(e)}")
        
    return articles_with_summaries


def save_articles_with_summaries(articles: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save articles with summaries to a JSON file.
    
    Args:
        articles: List of article dictionaries with summaries
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(articles, f, indent=2)
        
    print(f"Saved {len(articles)} articles with summaries to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate summaries using a fine-tuned model")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Input JSON file containing processed articles")
    input_group.add_argument("--dataset", type=str, help="Name of a Hugging Face dataset")
    
    # Hugging Face dataset options
    parser.add_argument("--hf_subset", type=str, help="Subset/config of the Hugging Face dataset")
    parser.add_argument("--hf_split", type=str, default="test", help="Split to use for the Hugging Face dataset")
    parser.add_argument("--text-column", type=str, default="text", help="Column name for source texts")
    parser.add_argument("--limit", type=int, help="Maximum number of examples to process")
    
    # Output and model options
    parser.add_argument("--output", type=str, help="Output JSON file for articles with summaries")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), 
                        help="Directory containing the fine-tuned model")
    parser.add_argument("--device", type=int, default=-1, help="Device to use (-1 for CPU, 0+ for specific GPU)")
    parser.add_argument("--max-length", type=int, default=150, help="Maximum length of the summary in tokens")
    parser.add_argument("--min-length", type=int, default=40, help="Minimum length of the summary in tokens")
    args = parser.parse_args()
    
    # Load input data
    if args.dataset:
        # Load from Hugging Face
        articles = load_huggingface_dataset(
            dataset_name=args.dataset,
            split=args.hf_split,
            subset=args.hf_subset,
            text_column=args.text_column,
            limit=args.limit
        )
        
        # Set default output file if not provided
        if not args.output:
            
            hf_name = args.dataset.replace("/", "_")
            args.output = str(DATA_DIR / f"{hf_name}_finetuned_summaries.json")
    else:
        # Load from local JSON file
        articles = load_articles(args.input)
        
        # Set default output file if not provided
        if not args.output:
            input_path = Path(args.input)
            args.output = str(DATA_DIR / f"{input_path.stem}_finetuned_summaries.json")
    
    # Set up summarization pipeline
    summarizer = setup_summarization_pipeline(args.model_dir, args.device)
    
    # Generate summaries
    articles_with_summaries = generate_summaries(
        articles,
        summarizer,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    # Save articles with summaries
    save_articles_with_summaries(articles_with_summaries, args.output)


if __name__ == "__main__":
    main() 