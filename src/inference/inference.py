#!/usr/bin/env python3
"""
Script to run inference with both baseline and fine-tuned summarization models.
Compares the performance of the two models on a test set of articles.
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from datasets import load_dataset
from peft import PeftModel, PeftConfig

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Default models
DEFAULT_BASELINE_MODEL = "facebook/bart-large-cnn"
DEFAULT_FINETUNED_PATH = "models/finetuned/yfinance-md"


def load_articles(input_file: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.
    
    Args:
        input_file: Path to the JSON file containing articles
        limit: Optional limit on the number of articles to load
        
    Returns:
        List of article dictionaries
    """
    with open(input_file, 'r') as f:
        articles = json.load(f)
    
    if limit is not None:
        articles = articles[:limit]
        
    print(f"Loaded {len(articles)} articles from {input_file}")
    return articles


def load_huggingface_dataset(
    dataset_name: str,
    text_column: str = "text",
    summary_column: str = "summary",
    split: str = "test",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load articles from a Hugging Face dataset.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        text_column: Column name for the article text
        summary_column: Column name for the summary (if available)
        split: Dataset split to use (e.g., "test", "validation")
        limit: Optional limit on the number of articles to load
        
    Returns:
        List of article dictionaries
    """
    print(f"Loading Hugging Face dataset: {dataset_name}")
    
    # Try to load the dataset
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(dataset)} examples from {dataset_name} ({split} split)")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Trying to load without specifying a split...")
        try:
            # Some datasets have different split names
            dataset_dict = load_dataset(dataset_name)
            
            # Try to find a test or validation split
            if "test" in dataset_dict:
                dataset = dataset_dict["test"]
            elif "validation" in dataset_dict:
                dataset = dataset_dict["validation"]
            else:
                # Use whatever split is available
                split_name = list(dataset_dict.keys())[0]
                dataset = dataset_dict[split_name]
                
            print(f"Loaded {len(dataset)} examples from {dataset_name} ({split_name} split)")
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            return []
    
    # Map column names if needed
    if text_column not in dataset.features:
        # Try to find text column
        potential_text_columns = ["text", "article", "document", "content", "input", "source"]
        for col in potential_text_columns:
            if col in dataset.features:
                print(f"Using '{col}' as text column instead of '{text_column}'")
                text_column = col
                break
        else:
            print(f"Could not find text column. Available columns: {list(dataset.features.keys())}")
            return []
    
    # Check if summary column exists
    has_summary = False
    if summary_column in dataset.features:
        has_summary = True
    else:
        # Try to find summary column
        potential_summary_columns = ["summary", "highlights", "target", "output", "headline", "title"]
        for col in potential_summary_columns:
            if col in dataset.features:
                print(f"Using '{col}' as summary column instead of '{summary_column}'")
                summary_column = col
                has_summary = True
                break
        else:
            print(f"Could not find summary column. Available columns: {list(dataset.features.keys())}")
    
    # Convert dataset to list of dictionaries
    articles = []
    
    for i, example in enumerate(dataset):
        if limit is not None and i >= limit:
            break
            
        article = {}
        
        # Extract text
        if text_column in example:
            article["text"] = example[text_column]
            
            # Add title if available
            if "title" in example and example["title"] != example[text_column]:
                article["title"] = example["title"]
        else:
            continue  # Skip if no text
            
        # Extract summary if available
        if has_summary and summary_column in example:
            article["reference_summary"] = example[summary_column]
            
        # Add metadata if available
        if "id" in example:
            article["id"] = example["id"]
            
        if "date" in example:
            article["date"] = example["date"]
            
        if "url" in example:
            article["url"] = example["url"]
            
        if "source" in example:
            article["source"] = example["source"]
            
        # Add to list
        articles.append(article)
    
    print(f"Converted {len(articles)} examples to article format")
    return articles


def setup_baseline_model(model_name: str, device: int = -1) -> pipeline:
    """
    Set up the baseline summarization model.
    
    Args:
        model_name: Name of the pre-trained model to use
        device: Device to use (-1 for CPU, 0+ for specific GPU)
        
    Returns:
        Hugging Face pipeline for summarization
    """
    print(f"Loading baseline model: {model_name}")
    
    # Check if CUDA is available and set device accordingly
    if torch.cuda.is_available() and device >= 0:
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU for inference")
        device = -1
    
    # Create pipeline
    summarizer = pipeline(
        "summarization",
        model=model_name,
        device=device
    )
    
    return summarizer


def setup_finetuned_model(model_path: str, device: int = -1) -> pipeline:
    """
    Set up the fine-tuned summarization model.
    
    Args:
        model_path: Path to the fine-tuned model
        device: Device to use (-1 for CPU, 0+ for specific GPU)
        
    Returns:
        Hugging Face pipeline for summarization
    """
    print(f"Loading fine-tuned model from: {model_path}")
    
    # Check if CUDA is available and set device accordingly
    if torch.cuda.is_available() and device >= 0:
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        device_str = f"cuda:{device}"
    else:
        print("Using CPU for inference")
        device_str = "cpu"
        device = -1
    
    # Load config, model, and tokenizer
    config = PeftConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    # Load the base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        device_map=device_str if device >= 0 else None
    )
    
    # Load the PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Create pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    return summarizer


def summarize_articles(
    articles: List[Dict[str, Any]],
    baseline_model,
    finetuned_model,
    max_length: int = 150,
    min_length: int = 40
) -> List[Dict[str, Any]]:
    """
    Generate summaries for articles using both baseline and fine-tuned models.
    
    Args:
        articles: List of article dictionaries
        baseline_model: Baseline summarization pipeline
        finetuned_model: Fine-tuned summarization pipeline
        max_length: Maximum length of the summary in tokens
        min_length: Minimum length of the summary in tokens
        
    Returns:
        List of article dictionaries with summaries added
    """
    articles_with_summaries = []
    
    for article in tqdm(articles, desc="Generating summaries"):
        # Get the text to summarize (prefer cleaned_text if available)
        text = article.get("cleaned_text", article.get("text", ""))
        
        # Skip articles without text
        if not text:
            continue
            
        article_with_summary = article.copy()
        
        # Generate baseline summary
        try:
            start_time = time.time()
            baseline_result = baseline_model(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            baseline_time = time.time() - start_time
            
            article_with_summary["baseline_summary"] = baseline_result[0]["summary_text"]
            article_with_summary["baseline_time"] = baseline_time
            
        except Exception as e:
            print(f"Error generating baseline summary: {str(e)}")
            article_with_summary["baseline_summary"] = ""
            article_with_summary["baseline_time"] = 0
        
        # Generate fine-tuned summary
        try:
            start_time = time.time()
            finetuned_result = finetuned_model(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            finetuned_time = time.time() - start_time
            
            article_with_summary["finetuned_summary"] = finetuned_result[0]["summary_text"]
            article_with_summary["finetuned_time"] = finetuned_time
            
        except Exception as e:
            print(f"Error generating fine-tuned summary: {str(e)}")
            article_with_summary["finetuned_summary"] = ""
            article_with_summary["finetuned_time"] = 0
        
        articles_with_summaries.append(article_with_summary)
        
    return articles_with_summaries


def save_results(articles: List[Dict[str, Any]], output_file: str) -> None:
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
    parser = argparse.ArgumentParser(description="Run inference with baseline and fine-tuned summarization models")
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Input JSON file containing articles")
    input_group.add_argument("--dataset", type=str, help="Hugging Face dataset name to use")
    
    # Dataset options (only used if --dataset is specified)
    parser.add_argument("--text-column", type=str, default="text",
                      help="Column name for article text in the dataset")
    parser.add_argument("--summary-column", type=str, default="summary",
                      help="Column name for summary in the dataset")
    parser.add_argument("--split", type=str, default="test",
                      help="Dataset split to use (e.g., 'test', 'validation')")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output JSON file for articles with summaries")
    
    # Model options
    parser.add_argument("--baseline-model", type=str, default=DEFAULT_BASELINE_MODEL,
                        help="Baseline pre-trained model to use")
    parser.add_argument("--finetuned-model", type=str, default=DEFAULT_FINETUNED_PATH,
                        help="Path to the fine-tuned model")
    parser.add_argument("--device", type=int, default=-1,
                        help="Device to use (-1 for CPU, 0+ for specific GPU)")
    
    # Summarization options
    parser.add_argument("--max-length", type=int, default=150,
                        help="Maximum length of the summary in tokens")
    parser.add_argument("--min-length", type=int, default=40,
                        help="Minimum length of the summary in tokens")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of articles to process")
    
    # Filtering options
    parser.add_argument("--ticker", type=str, help="Filter articles by ticker")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output:
        if args.input:
            input_path = Path(args.input)
            args.output = str(DATA_DIR / f"{input_path.stem}_comparison.json")
        else:
            # Create output file name based on dataset
            dataset_name = args.dataset.split("/")[-1]
            args.output = str(DATA_DIR / f"{dataset_name}_comparison.json")
    
    # Load articles
    if args.input:
        articles = load_articles(args.input, args.limit)
    else:
        articles = load_huggingface_dataset(
            args.dataset,
            text_column=args.text_column,
            summary_column=args.summary_column,
            split=args.split,
            limit=args.limit
        )
    
    # Check if we have articles to process
    if not articles:
        print("No articles to process. Exiting.")
        return
    
    # Filter by ticker if specified
    if args.ticker:
        ticker = args.ticker.upper()
        articles = [article for article in articles if article.get("ticker") == ticker]
        print(f"Filtered to {len(articles)} articles for ticker {ticker}")
    
    # Set up models
    baseline_model = setup_baseline_model(args.baseline_model, args.device)
    finetuned_model = setup_finetuned_model(args.finetuned_model, args.device)
    
    # Generate summaries
    articles_with_summaries = summarize_articles(
        articles,
        baseline_model,
        finetuned_model,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    # Save results
    save_results(articles_with_summaries, args.output)
    
    # Print comparison
    print("\nSummary Comparison Examples:")
    for i, article in enumerate(articles_with_summaries[:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"Title: {article.get('title', 'N/A')}")
        print("\nBaseline Summary:")
        print(article.get("baseline_summary", "N/A"))
        print(f"(Generated in {article.get('baseline_time', 0):.2f} seconds)")
        print("\nFine-tuned Summary:")
        print(article.get("finetuned_summary", "N/A"))
        print(f"(Generated in {article.get('finetuned_time', 0):.2f} seconds)")
        
        # Print reference summary if available
        if "reference_summary" in article:
            print("\nReference Summary:")
            print(article.get("reference_summary", "N/A"))
            
        print("="*80)


if __name__ == "__main__":
    main() 