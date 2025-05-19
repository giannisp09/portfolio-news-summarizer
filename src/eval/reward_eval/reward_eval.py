#!/usr/bin/env python3
"""
Script to evaluate summary quality using a reward model.
Compares baseline and fine-tuned model summaries.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
import torch

# Import the reward model evaluator
from eval.reward_eval.reward_model_eval import RewardModelEvaluator, get_best_device

# Ensure data and results directories exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_articles(input_file: str) -> List[Dict[str, Any]]:
    """
    Load articles with summaries from a JSON file.
    
    Args:
        input_file: Path to the JSON file containing articles with summaries
        
    Returns:
        List of article dictionaries with summaries
    """
    with open(input_file, 'r') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {input_file}")
    return articles


def evaluate_summaries(
    articles: List[Dict[str, Any]], 
    evaluator: RewardModelEvaluator,
    text_key: str = "text",
    summaries_to_evaluate: List[str] = ["baseline_summary", "finetuned_summary"]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate summaries using the reward model.
    
    Args:
        articles: List of article dictionaries with summaries
        evaluator: Initialized RewardModelEvaluator
        text_key: Key for the source text
        summaries_to_evaluate: List of keys for summaries to evaluate
        
    Returns:
        Dictionary with evaluation results for each summary type
    """
    print("Evaluating summaries with reward model...")
    
    # Initialize results dictionary
    results = {}
    for summary_key in summaries_to_evaluate:
        results[summary_key] = {
            "scores": []
        }
    
    # Evaluate each summary
    for article in tqdm(articles, desc="Reward model evaluation"):
        text = article.get(text_key)
        
        # Skip articles without text
        if not text:
            continue
            
        # Evaluate each summary type
        for summary_key in summaries_to_evaluate:
            summary = article.get(summary_key)
            
            # Skip if summary doesn't exist
            if not summary:
                continue
                
            # Get reward score
            score = evaluator.score(text, summary)
            results[summary_key]["scores"].append(score)
    
    # Calculate statistics for each summary type
    for summary_key in summaries_to_evaluate:
        if results[summary_key]["scores"]:
            scores = results[summary_key]["scores"]
            results[summary_key]["mean"] = np.mean(scores)
            results[summary_key]["std"] = np.std(scores)
            results[summary_key]["min"] = np.min(scores)
            results[summary_key]["max"] = np.max(scores)
            results[summary_key]["median"] = np.median(scores)
            results[summary_key]["count"] = len(scores)
    
    return results


def generate_plots(results: Dict[str, Dict[str, Any]], output_prefix: str = "reward_model_eval"):
    """
    Generate plots comparing reward model scores.
    
    Args:
        results: Dictionary with evaluation results
        output_prefix: Prefix for output files
    """
    plt.figure(figsize=(12, 6))
    
    # Box plot
    plt.subplot(1, 2, 1)
    data = []
    labels = []
    
    for summary_key, summary_results in results.items():
        if "scores" in summary_results and summary_results["scores"]:
            data.append(summary_results["scores"])
            labels.append(summary_key.replace("_summary", ""))
    
    sns.boxplot(data=data)
    plt.xticks(range(len(labels)), labels)
    plt.title("Reward Model Scores Distribution")
    plt.ylabel("Score")
    
    # Bar plot with error bars
    plt.subplot(1, 2, 2)
    means = [results[k]["mean"] for k in results if "mean" in results[k]]
    stds = [results[k]["std"] for k in results if "std" in results[k]]
    
    plt.bar(range(len(labels)), means, yerr=stds, capsize=10, alpha=0.7)
    plt.xticks(range(len(labels)), labels)
    plt.title("Average Reward Model Scores")
    plt.ylabel("Mean Score")
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_comparison.png")
    print(f"Saved comparison plot to {output_prefix}_comparison.png")
    
    # Histogram
    plt.figure(figsize=(12, 6))
    for i, (summary_key, summary_results) in enumerate(results.items()):
        if "scores" in summary_results and summary_results["scores"]:
            label = summary_key.replace("_summary", "")
            plt.hist(summary_results["scores"], bins=15, alpha=0.5, label=label)
    
    plt.title("Distribution of Reward Model Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_histogram.png")
    print(f"Saved histogram to {output_prefix}_histogram.png")


def generate_report(
    articles: List[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
    output_file: str = "reward_model_eval_report.json"
) -> None:
    """
    Generate a report with summary statistics.
    
    Args:
        articles: List of article dictionaries with summaries
        results: Dictionary with evaluation results
        output_file: Path to output JSON file
    """
    # Create report dictionary
    report = {
        "meta": {
            "num_articles": len(articles),
            "evaluated_at": pd.Timestamp.now().isoformat()
        },
        "reward_model_scores": {}
    }
    
    # Add results for each summary type
    for summary_key, summary_results in results.items():
        if "mean" in summary_results:
            report["reward_model_scores"][summary_key] = {
                "mean": summary_results["mean"],
                "std": summary_results["std"],
                "min": summary_results["min"],
                "max": summary_results["max"],
                "median": summary_results["median"],
                "count": summary_results["count"]
            }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved evaluation report to {output_file}")
    
    # Print summary
    print("\n=== REWARD MODEL EVALUATION SUMMARY ===")
    for summary_key, summary_results in results.items():
        if "mean" in summary_results:
            name = summary_key.replace("_summary", "").capitalize()
            print(f"{name} Mean Score: {summary_results['mean']:.4f} ± {summary_results['std']:.4f} (n={summary_results['count']})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries using a reward model")
    
    # Input options
    parser.add_argument("--input", type=str, required=True, help="Input JSON file containing articles with summaries")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the reward model")
    
    # Column options
    parser.add_argument("--text-column", type=str, default="text", help="Column name for source texts")
    parser.add_argument("--summaries", type=str, nargs="+", default=["baseline_summary", "finetuned_summary"], 
                       help="Summary column names to evaluate")
    
    # Output options
    parser.add_argument("--output-prefix", type=str, default="reward_model_eval", 
                       help="Prefix for output files")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), 
                       help="Directory for output files")
    
    # Model options
    parser.add_argument("--device", type=int, default=-1, help="Device to use (-1 for auto-detect, 0+ for specific GPU)")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenization")
    
    args = parser.parse_args()
    
    # Load articles
    articles = load_articles(args.input)
    
    # Initialize reward model evaluator
    evaluator = RewardModelEvaluator(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length
    )
    
    # Evaluate summaries
    results = evaluate_summaries(
        articles=articles,
        evaluator=evaluator,
        text_key=args.text_column,
        summaries_to_evaluate=args.summaries
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set output paths
    output_prefix = f"{args.output_dir}/{args.output_prefix}"
    output_report = f"{args.output_dir}/{args.output_prefix}_report.json"
    
    # Generate plots
    generate_plots(results, output_prefix)
    
    # Generate report
    generate_report(articles, results, output_report)


if __name__ == "__main__":
    main() 