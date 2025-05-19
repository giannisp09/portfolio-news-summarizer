#!/usr/bin/env python3
"""
Script to prepare training data for fine-tuning summarization models.
- Can use existing datasets from Hugging Face
- Can use GPT-4 or other LLMs to generate "silver" summaries
- Creates training, validation, and test splits
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import csv
import requests

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Constants
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


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


def download_cnn_dailymail_dataset(subset_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Download a subset of the CNN/DailyMail dataset from Hugging Face.
    
    Args:
        subset_size: Number of examples to download
        
    Returns:
        List of article-summary pairs
    """
    print(f"Downloading CNN/DailyMail dataset (subset of {subset_size} examples)...")
    
    # Load the dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Get the training split
    train_data = dataset["train"]
    
    # Randomly sample a subset
    indices = random.sample(range(len(train_data)), min(subset_size, len(train_data)))
    subset = [train_data[i] for i in indices]
    
    # Convert to our format
    article_summary_pairs = []
    for item in subset:
        pair = {
            "ticker": "NEWS",  # Generic ticker for non-stock news
            "title": "",  # CNN/DailyMail doesn't always have titles
            "text": item["article"],
            "summary": item["highlights"],
            "source": "CNN/DailyMail Dataset",
            "url": ""
        }
        article_summary_pairs.append(pair)
    
    return article_summary_pairs


def generate_gpt_summaries(articles: List[Dict[str, Any]], api_key: str, model: str = DEFAULT_OPENAI_MODEL) -> List[Dict[str, Any]]:
    """
    Generate summaries using OpenAI's GPT models.
    
    Args:
        articles: List of article dictionaries
        api_key: OpenAI API key
        model: OpenAI model to use
        
    Returns:
        List of articles with GPT-generated summaries
    """
    if not api_key:
        raise ValueError("OpenAI API key not provided")
    
    print(f"Generating summaries with {model}...")
    
    articles_with_summaries = []
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    for article in tqdm(articles, desc=f"Generating {model} summaries"):
        text = article.get("cleaned_text", article.get("text", ""))
        title = article.get("title", "")
        
        if not text:
            continue
        
        # Prepare the prompt
        prompt = f"""Please create a concise, factual summary of the following news article about {article.get('ticker', '')}. 
Title: {title}

Article:
{text[:4000]}  # Truncate to avoid token limits

Summary:"""

        # Request data
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 250
        }
        
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            
            # Extract summary
            summary = response.json()["choices"][0]["message"]["content"].strip()
            
            # Add summary to article
            article_with_summary = article.copy()
            article_with_summary["gpt_summary"] = summary
            
            articles_with_summaries.append(article_with_summary)
            
        except Exception as e:
            print(f"Error generating summary for article '{title}': {str(e)}")
    
    return articles_with_summaries


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data: List of article dictionaries with summaries
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle data
    random.shuffle(data)
    
    # Calculate split indices
    n = len(data)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)
    
    # Split data
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]
    
    print(f"Split data into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test examples")
    
    return train_data, val_data, test_data


def save_data_for_finetuning(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    output_prefix: str,
    summary_field: str
) -> None:
    """
    Save data splits for fine-tuning.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        output_prefix: Prefix for output files
        summary_field: Field name containing the summary to use
    """
    # Function to convert to fine-tuning format
    def to_finetuning_format(data):
        return [
            {
                "text": item.get("cleaned_text", item.get("text", "")),
                "summary": item.get(summary_field, "")
            }
            for item in data
            if item.get(summary_field)
        ]
    
    # Convert data
    train_ft = to_finetuning_format(train_data)
    val_ft = to_finetuning_format(val_data)
    test_ft = to_finetuning_format(test_data)
    
    # Save as JSON
    with open(f"{output_prefix}_train.json", 'w') as f:
        json.dump(train_ft, f, indent=2)
    
    with open(f"{output_prefix}_val.json", 'w') as f:
        json.dump(val_ft, f, indent=2)
    
    with open(f"{output_prefix}_test.json", 'w') as f:
        json.dump(test_ft, f, indent=2)
    
    # Save as CSV for easier loading
    for split_name, split_data in [("train", train_ft), ("val", val_ft), ("test", test_ft)]:
        df = pd.DataFrame(split_data)
        df.to_csv(f"{output_prefix}_{split_name}.csv", index=False, quoting=csv.QUOTE_ALL)
    
    print(f"Saved {len(train_ft)} training, {len(val_ft)} validation, and {len(test_ft)} test examples")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning summarization models")
    parser.add_argument("--input", type=str, help="Input JSON file containing processed articles")
    parser.add_argument("--output-prefix", type=str, default=str(DATA_DIR / "finetune"),
                        help="Prefix for output files")
    parser.add_argument("--source", choices=["cnn_dailymail", "articles", "both"], default="both",
                       help="Source of training data: CNN/DailyMail dataset, articles with GPT summaries, or both")
    parser.add_argument("--subset-size", type=int, default=1000,
                        help="Number of examples to download from CNN/DailyMail")
    parser.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL,
                        help="OpenAI model to use for generating summaries")
    parser.add_argument("--summary-field", type=str, default="gpt_summary",
                        help="Field name containing the summary to use (gpt_summary or baseline_summary)")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                        help="Ratio of training data")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help="Ratio of validation data")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO,
                        help="Ratio of test data")
    args = parser.parse_args()
    
    all_data = []
    
    # Get data from CNN/DailyMail dataset
    if args.source in ["cnn_dailymail", "both"]:
        cnn_dailymail_data = download_cnn_dailymail_dataset(args.subset_size)
        all_data.extend(cnn_dailymail_data)
    
    # Get data from articles with GPT summaries
    if args.source in ["articles", "both"] and args.input:
        # Load articles
        articles = load_articles(args.input)
        
        # Generate summaries if needed
        if args.summary_field == "gpt_summary" and not articles[0].get("gpt_summary"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            articles = generate_gpt_summaries(articles, api_key, args.openai_model)
        
        all_data.extend(articles)
    
    # Split data
    train_data, val_data, test_data = split_data(
        all_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Save data for fine-tuning
    save_data_for_finetuning(
        train_data,
        val_data,
        test_data,
        args.output_prefix,
        args.summary_field
    )


if __name__ == "__main__":
    main() 