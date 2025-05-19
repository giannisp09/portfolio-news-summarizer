#!/usr/bin/env python3
"""
Script to generate baseline summaries using Hugging Face's summarization pipeline.
Uses pre-trained models without fine-tuning.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Define default model
DEFAULT_MODEL = "facebook/bart-large-cnn"


def load_articles(input_file: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load processed articles from a JSON file.
    
    Args:
        input_file: Path to the JSON file containing processed articles
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


def setup_summarization_pipeline(model_name: str, device: int = -1) -> pipeline:
    """
    Set up the Hugging Face summarization pipeline.
    
    Args:
        model_name: Name of the pre-trained model to use
        device: Device to use (-1 for CPU, 0+ for specific GPU)
        
    Returns:
        Hugging Face pipeline for summarization
    """
    print(f"Loading summarization model: {model_name}")
    
    # Check if CUDA is available and set device accordingly
    if torch.cuda.is_available() and device >= 0:
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU for inference")
        device = -1

    # check for MPS
    if torch.backends.mps.is_available():
        print("Using MPS for inference")
        device = "mps"
    else:
        print("Using CPU for inference")
        device = -1
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Create pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    return summarizer


def generate_summaries(
    articles: List[Dict[str, Any]],
    summarizer,
    max_length: int = 512,
    min_length: int = 40
) -> List[Dict[str, Any]]:
    """
    Generate summaries for each article.
    
    Args:
        articles: List of article dictionaries
        summarizer: Hugging Face summarization pipeline
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
            
        # Generate summary
        try:
            start_time = time.time()
            summary = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            generation_time = time.time() - start_time
            
            # Add summary to article
            article_with_summary = article.copy()
            article_with_summary["baseline_summary"] = summary[0]["summary_text"]
            article_with_summary["baseline_time"] = generation_time
            
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
    parser = argparse.ArgumentParser(description="Generate baseline summaries using pre-trained models")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Input JSON file containing processed articles")
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
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face model to use for summarization")
    parser.add_argument("--device", type=int, default=-1, help="Device to use (-1 for CPU, 0+ for specific GPU)")
    parser.add_argument("--max-length", type=int, default=150, help="Maximum length of the summary in tokens")
    parser.add_argument("--min-length", type=int, default=40, help="Minimum length of the summary in tokens")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of articles to process")
    
    # Filtering options
    parser.add_argument("--ticker", type=str, help="Filter articles by ticker")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output:
        if args.input:
            input_path = Path(args.input)
            args.output = str(DATA_DIR / f"{input_path.stem}_baseline_summaries.json")
        else:
            # Create output file name based on dataset
            dataset_name = args.dataset.split("/")[-1]
            args.output = str(DATA_DIR / f"{dataset_name}_baseline_summaries.json")
    
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
    
    # Set up summarization pipeline
    summarizer = setup_summarization_pipeline(args.model, args.device)
    
    # Generate summaries
    articles_with_summaries = generate_summaries(
        articles,
        summarizer,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    # Save articles with summaries
    save_articles_with_summaries(articles_with_summaries, args.output)
    
    # Print example summaries
    print("\nSummary Examples:")
    for i, article in enumerate(articles_with_summaries[:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"Title: {article.get('title', 'N/A')}")
        print("\nBaseline Summary:")
        print(article.get("baseline_summary", "N/A"))
        print(f"(Generated in {article.get('baseline_time', 0):.2f} seconds)")
        
        # Print reference summary if available
        if "reference_summary" in article:
            print("\nReference Summary:")
            print(article.get("reference_summary", "N/A"))
            
        print("="*80)


if __name__ == "__main__":
    main() 