#!/usr/bin/env python3
"""
Script to evaluate and compare causal LLM summarization performance.
Compares baseline instruction-tuned LLMs with fine-tuned versions.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Set up matplotlib style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_articles(input_file: str) -> List[Dict[str, Any]]:
    """
    Load processed articles with summaries from a JSON file.
    
    Args:
        input_file: Path to the JSON file containing articles with summaries
        
    Returns:
        List of article dictionaries
    """
    with open(input_file, 'r') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {input_file}")
    
    # Check if articles have the expected summary fields
    has_baseline_llm = any("llm_summary" in article for article in articles)
    has_finetuned_llm = any("finetuned_llm_summary" in article for article in articles)
    
    if not has_baseline_llm and not has_finetuned_llm:
        print("Warning: No causal LLM summaries found in the dataset.")
        print("Expected fields 'llm_summary' or 'finetuned_llm_summary'.")
    
    return articles


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    return nltk.word_tokenize(text.lower())


def compute_rouge_scores(
    baseline_summaries: List[str], 
    finetuned_summaries: List[str],
    reference_summaries: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE scores for baseline and fine-tuned summaries.
    
    Args:
        baseline_summaries: List of baseline summaries
        finetuned_summaries: List of fine-tuned summaries
        reference_summaries: Optional list of reference summaries
            If not provided, fine-tuned summaries are compared to baseline
        
    Returns:
        Dictionary with ROUGE scores
    """
    scores = {"baseline": {}, "finetuned": {}}
    
    # Initialize ROUGE scorer with multiple variants
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Check if we have both baseline and finetuned summaries
    has_baseline = len(baseline_summaries) > 0
    has_finetuned = len(finetuned_summaries) > 0
    
    if not has_baseline and not has_finetuned:
        print("No summaries available for ROUGE calculation")
        return scores
    
    # If reference summaries are provided, compare both to references
    if reference_summaries:
        if has_baseline:
            baseline_rouge1, baseline_rouge2, baseline_rougeL = 0, 0, 0
            baseline_count = 0
            
            for baseline, reference in tqdm(
                zip(baseline_summaries, reference_summaries),
                desc="Computing ROUGE scores for baseline",
                total=len(baseline_summaries)
            ):
                # Skip empty summaries or references
                if not baseline or not reference:
                    continue
                    
                # Compute scores for baseline vs reference
                baseline_scores = scorer.score(reference, baseline)
                baseline_rouge1 += baseline_scores['rouge1'].fmeasure
                baseline_rouge2 += baseline_scores['rouge2'].fmeasure
                baseline_rougeL += baseline_scores['rougeL'].fmeasure
                baseline_count += 1
            
            # Calculate averages for baseline
            if baseline_count > 0:
                scores["baseline"]["rouge1"] = baseline_rouge1 / baseline_count
                scores["baseline"]["rouge2"] = baseline_rouge2 / baseline_count
                scores["baseline"]["rougeL"] = baseline_rougeL / baseline_count
        
        if has_finetuned:
            finetuned_rouge1, finetuned_rouge2, finetuned_rougeL = 0, 0, 0
            finetuned_count = 0
            
            for finetuned, reference in tqdm(
                zip(finetuned_summaries, reference_summaries),
                desc="Computing ROUGE scores for fine-tuned",
                total=len(finetuned_summaries)
            ):
                # Skip empty summaries or references
                if not finetuned or not reference:
                    continue
                    
                # Compute scores for fine-tuned vs reference
                finetuned_scores = scorer.score(reference, finetuned)
                finetuned_rouge1 += finetuned_scores['rouge1'].fmeasure
                finetuned_rouge2 += finetuned_scores['rouge2'].fmeasure
                finetuned_rougeL += finetuned_scores['rougeL'].fmeasure
                finetuned_count += 1
            
            # Calculate averages for fine-tuned
            if finetuned_count > 0:
                scores["finetuned"]["rouge1"] = finetuned_rouge1 / finetuned_count
                scores["finetuned"]["rouge2"] = finetuned_rouge2 / finetuned_count
                scores["finetuned"]["rougeL"] = finetuned_rougeL / finetuned_count
        
    elif has_baseline and has_finetuned:  
        # Without reference, compare fine-tuned summaries to baseline
        finetuned_rouge1, finetuned_rouge2, finetuned_rougeL = 0, 0, 0
        count = 0
        
        for baseline, finetuned in tqdm(
            zip(baseline_summaries, finetuned_summaries),
            desc="Computing ROUGE scores",
            total=min(len(baseline_summaries), len(finetuned_summaries))
        ):
            # Skip empty summaries
            if not baseline or not finetuned:
                continue
                
            # Compute scores for fine-tuned vs baseline
            finetuned_scores = scorer.score(baseline, finetuned)
            finetuned_rouge1 += finetuned_scores['rouge1'].fmeasure
            finetuned_rouge2 += finetuned_scores['rouge2'].fmeasure
            finetuned_rougeL += finetuned_scores['rougeL'].fmeasure
            count += 1
        
        # Calculate averages
        if count > 0:
            # Baseline compared to itself would be 1.0
            scores["baseline"]["rouge1"] = 1.0
            scores["baseline"]["rouge2"] = 1.0
            scores["baseline"]["rougeL"] = 1.0
            
            scores["finetuned"]["rouge1"] = finetuned_rouge1 / count
            scores["finetuned"]["rouge2"] = finetuned_rouge2 / count
            scores["finetuned"]["rougeL"] = finetuned_rougeL / count
    
    return scores


def compute_bertscore(
    baseline_summaries: List[str], 
    finetuned_summaries: List[str],
    reference_summaries: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute BERTScores for summaries.
    
    Args:
        baseline_summaries: List of baseline summaries
        finetuned_summaries: List of fine-tuned summaries
        reference_summaries: Optional list of reference summaries
            If not provided, fine-tuned summaries are compared to baseline
            
    Returns:
        Dictionary with BERTScores
    """
    scores = {"baseline": {}, "finetuned": {}}
    
    # Check if we have baseline and finetuned summaries
    has_baseline = len(baseline_summaries) > 0
    has_finetuned = len(finetuned_summaries) > 0
    
    if not has_baseline and not has_finetuned:
        print("No summaries available for BERTScore calculation")
        return scores
    
    # If reference summaries are provided, compare available summaries to references
    if reference_summaries:
        if has_baseline:
            # Filter out empty summaries
            valid_baseline = []
            valid_baseline_refs = []
            
            for baseline, reference in zip(baseline_summaries, reference_summaries):
                if baseline and reference:
                    valid_baseline.append(baseline)
                    valid_baseline_refs.append(reference)
            
            if valid_baseline:
                print("Computing BERTScore for baseline summaries...")
                baseline_P, baseline_R, baseline_F1 = bert_score(
                    valid_baseline, valid_baseline_refs, lang="en", verbose=True
                )
                scores["baseline"]["bertscore"] = baseline_F1.mean().item()
        
        if has_finetuned:
            # Filter out empty summaries
            valid_finetuned = []
            valid_finetuned_refs = []
            
            for finetuned, reference in zip(finetuned_summaries, reference_summaries):
                if finetuned and reference:
                    valid_finetuned.append(finetuned)
                    valid_finetuned_refs.append(reference)
            
            if valid_finetuned:
                print("Computing BERTScore for fine-tuned summaries...")
                finetuned_P, finetuned_R, finetuned_F1 = bert_score(
                    valid_finetuned, valid_finetuned_refs, lang="en", verbose=True
                )
                scores["finetuned"]["bertscore"] = finetuned_F1.mean().item()
        
    elif has_baseline and has_finetuned:
        # Without reference, compare fine-tuned summaries to baseline
        valid_baseline = []
        valid_finetuned = []
        
        for baseline, finetuned in zip(baseline_summaries, finetuned_summaries):
            if baseline and finetuned:
                valid_baseline.append(baseline)
                valid_finetuned.append(finetuned)
        
        if valid_baseline and valid_finetuned:
            print("Computing BERTScore comparing fine-tuned to baseline summaries...")
            _, _, finetuned_F1 = bert_score(
                valid_finetuned, valid_baseline, lang="en", verbose=True
            )
            
            scores["baseline"]["bertscore"] = 1.0  # Perfect match with itself
            scores["finetuned"]["bertscore"] = finetuned_F1.mean().item()
    
    return scores


def compute_bleu_scores(
    baseline_summaries: List[str], 
    finetuned_summaries: List[str],
    reference_summaries: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute BLEU scores for summaries.
    
    Args:
        baseline_summaries: List of baseline summaries
        finetuned_summaries: List of fine-tuned summaries
        reference_summaries: Optional list of reference summaries
            If not provided, fine-tuned summaries are compared to baseline
            
    Returns:
        Dictionary with BLEU scores
    """
    scores = {"baseline": {}, "finetuned": {}}
    
    # Check if we have baseline and finetuned summaries
    has_baseline = len(baseline_summaries) > 0
    has_finetuned = len(finetuned_summaries) > 0
    
    if not has_baseline and not has_finetuned:
        print("No summaries available for BLEU calculation")
        return scores
    
    # Initialize smoothing function for BLEU
    smoother = SmoothingFunction().method1
    
    # If reference summaries are provided, compare available summaries to references
    if reference_summaries:
        if has_baseline:
            baseline_bleu = 0.0
            baseline_count = 0
            
            for baseline, reference in tqdm(
                zip(baseline_summaries, reference_summaries),
                desc="Computing BLEU scores for baseline",
                total=len(baseline_summaries)
            ):
                # Skip empty summaries
                if not baseline or not reference:
                    continue
                    
                # Tokenize
                baseline_tokens = tokenize_text(baseline)
                reference_tokens = tokenize_text(reference)
                
                # Avoid empty tokens
                if not baseline_tokens or not reference_tokens:
                    continue
                
                # Compute scores
                try:
                    baseline_bleu += sentence_bleu(
                        [reference_tokens], baseline_tokens, smoothing_function=smoother
                    )
                    baseline_count += 1
                except Exception as e:
                    print(f"Error computing BLEU: {e}")
                    continue
            
            # Calculate average for baseline
            if baseline_count > 0:
                scores["baseline"]["bleu"] = baseline_bleu / baseline_count
        
        if has_finetuned:
            finetuned_bleu = 0.0
            finetuned_count = 0
            
            for finetuned, reference in tqdm(
                zip(finetuned_summaries, reference_summaries),
                desc="Computing BLEU scores for fine-tuned",
                total=len(finetuned_summaries)
            ):
                # Skip empty summaries
                if not finetuned or not reference:
                    continue
                    
                # Tokenize
                finetuned_tokens = tokenize_text(finetuned)
                reference_tokens = tokenize_text(reference)
                
                # Avoid empty tokens
                if not finetuned_tokens or not reference_tokens:
                    continue
                
                # Compute scores
                try:
                    finetuned_bleu += sentence_bleu(
                        [reference_tokens], finetuned_tokens, smoothing_function=smoother
                    )
                    finetuned_count += 1
                except Exception as e:
                    print(f"Error computing BLEU: {e}")
                    continue
            
            # Calculate average for fine-tuned
            if finetuned_count > 0:
                scores["finetuned"]["bleu"] = finetuned_bleu / finetuned_count
                
    elif has_baseline and has_finetuned:
        # Without reference, compare fine-tuned summaries to baseline
        finetuned_bleu = 0.0
        count = 0
        
        for baseline, finetuned in tqdm(
            zip(baseline_summaries, finetuned_summaries),
            desc="Computing BLEU scores",
            total=min(len(baseline_summaries), len(finetuned_summaries))
        ):
            # Skip empty summaries
            if not baseline or not finetuned:
                continue
                
            # Tokenize
            baseline_tokens = tokenize_text(baseline)
            finetuned_tokens = tokenize_text(finetuned)
            
            # Avoid empty tokens
            if not baseline_tokens or not finetuned_tokens:
                continue
            
            # Compute score for fine-tuned vs baseline
            try:
                finetuned_bleu += sentence_bleu(
                    [baseline_tokens], finetuned_tokens, smoothing_function=smoother
                )
                count += 1
            except Exception as e:
                print(f"Error computing BLEU: {e}")
                continue
        
        # Calculate averages
        if count > 0:
            # Baseline compared to itself would be 1.0
            scores["baseline"]["bleu"] = 1.0
            scores["finetuned"]["bleu"] = finetuned_bleu / count
    
    return scores


def visualize_results(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """
    Create visualizations for the evaluation results.
    
    Args:
        results: Results dictionary with metrics
        output_dir: Directory to save visualizations
    """
    # Create a directory for visualizations
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore', 'bleu']
    
    # Create a DataFrame for easier plotting
    data = []
    for metric in metrics:
        if metric in results["rouge"]["baseline"]:
            baseline_value = results["rouge"]["baseline"][metric]
            data.append({"Metric": metric, "Model": "Baseline LLM", "Score": baseline_value})
        
        if metric in results["rouge"]["finetuned"]:
            finetuned_value = results["rouge"]["finetuned"][metric]
            data.append({"Metric": metric, "Model": "Fine-tuned LLM", "Score": finetuned_value})
        
        elif metric in results["bertscore"]["baseline"]:
            baseline_value = results["bertscore"]["baseline"][metric]
            data.append({"Metric": metric, "Model": "Baseline LLM", "Score": baseline_value})
        
        if metric in results["bertscore"]["finetuned"]:
            finetuned_value = results["bertscore"]["finetuned"][metric]
            data.append({"Metric": metric, "Model": "Fine-tuned LLM", "Score": finetuned_value})
        
        elif metric in results["bleu"]["baseline"]:
            baseline_value = results["bleu"]["baseline"][metric]
            data.append({"Metric": metric, "Model": "Baseline LLM", "Score": baseline_value})
        
        if metric in results["bleu"]["finetuned"]:
            finetuned_value = results["bleu"]["finetuned"][metric]
            data.append({"Metric": metric, "Model": "Fine-tuned LLM", "Score": finetuned_value})
    
    if not data:
        print("No data available for visualization")
        return
        
    df = pd.DataFrame(data)
    
    # Bar chart of all metrics
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x="Metric", y="Score", hue="Model", data=df)
    plt.title("Summary Quality Metrics: Comparison")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, max(df["Score"].max() * 1.1, 1.0))  # Ensure upper limit is at least 1.0
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for p in chart.patches:
        height = p.get_height()
        chart.text(p.get_x() + p.get_width()/2, height + 0.01, f'{height:.3f}', 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(vis_dir / "metrics_comparison.png", dpi=300)
    plt.close()
    
    # Create a radar chart for the metrics if we have enough metrics and both models
    metrics_for_radar = []
    baseline_scores = []
    finetuned_scores = []
    
    has_both_models = False
    
    # Check if we have both models for any metric
    for metric in metrics:
        has_baseline = (metric in results["rouge"]["baseline"] or 
                        metric in results["bertscore"]["baseline"] or 
                        metric in results["bleu"]["baseline"])
        
        has_finetuned = (metric in results["rouge"]["finetuned"] or 
                         metric in results["bertscore"]["finetuned"] or 
                         metric in results["bleu"]["finetuned"])
        
        if has_baseline and has_finetuned:
            has_both_models = True
            break
    
    if has_both_models:
        for metric in metrics:
            baseline_value = None
            finetuned_value = None
            
            # Try to get values from different metric types
            if metric in results["rouge"]["baseline"]:
                baseline_value = results["rouge"]["baseline"][metric]
            elif metric in results["bertscore"]["baseline"]:
                baseline_value = results["bertscore"]["baseline"][metric]
            elif metric in results["bleu"]["baseline"]:
                baseline_value = results["bleu"]["baseline"][metric]
                
            if metric in results["rouge"]["finetuned"]:
                finetuned_value = results["rouge"]["finetuned"][metric]
            elif metric in results["bertscore"]["finetuned"]:
                finetuned_value = results["bertscore"]["finetuned"][metric]
            elif metric in results["bleu"]["finetuned"]:
                finetuned_value = results["bleu"]["finetuned"][metric]
            
            # Only add metrics that have both values
            if baseline_value is not None and finetuned_value is not None:
                metrics_for_radar.append(metric)
                baseline_scores.append(baseline_value)
                finetuned_scores.append(finetuned_value)
        
        # Create radar chart if we have at least 3 metrics
        if len(metrics_for_radar) >= 3:
            angles = np.linspace(0, 2*np.pi, len(metrics_for_radar), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            baseline_scores += baseline_scores[:1]  # Close the loop
            finetuned_scores += finetuned_scores[:1]  # Close the loop
            metrics_for_radar += [metrics_for_radar[0]]  # Close the loop
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Plot baseline
            ax.plot(angles, baseline_scores, 'b-', linewidth=2, label='Baseline LLM')
            ax.fill(angles, baseline_scores, 'b', alpha=0.1)
            
            # Plot fine-tuned
            ax.plot(angles, finetuned_scores, 'r-', linewidth=2, label='Fine-tuned LLM')
            ax.fill(angles, finetuned_scores, 'r', alpha=0.1)
            
            # Set ticks and labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_for_radar[:-1])
            
            # Add legend and title
            plt.legend(loc='upper right')
            plt.title('Summary Quality Comparison (Radar Chart)')
            
            plt.tight_layout()
            plt.savefig(vis_dir / "metrics_radar.png", dpi=300)
            plt.close()
    
    # Create a summary table image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    
    # Determine what columns to include
    has_baseline = any("baseline" in results[metric_type] and results[metric_type]["baseline"] 
                       for metric_type in results)
    has_finetuned = any("finetuned" in results[metric_type] and results[metric_type]["finetuned"] 
                        for metric_type in results)
    
    if has_baseline and has_finetuned:
        header = ['Metric', 'Baseline LLM', 'Fine-tuned LLM', 'Improvement']
    elif has_baseline:
        header = ['Metric', 'Baseline LLM']
    elif has_finetuned:
        header = ['Metric', 'Fine-tuned LLM']
    else:
        print("No data available for table visualization")
        return
    
    for metric in metrics:
        baseline = None
        finetuned = None
        
        # Try to get values from different metric types
        if metric in results.get("rouge", {}).get("baseline", {}):
            baseline = results["rouge"]["baseline"][metric]
        elif metric in results.get("bertscore", {}).get("baseline", {}):
            baseline = results["bertscore"]["baseline"][metric]
        elif metric in results.get("bleu", {}).get("baseline", {}):
            baseline = results["bleu"]["baseline"][metric]
            
        if metric in results.get("rouge", {}).get("finetuned", {}):
            finetuned = results["rouge"]["finetuned"][metric]
        elif metric in results.get("bertscore", {}).get("finetuned", {}):
            finetuned = results["bertscore"]["finetuned"][metric]
        elif metric in results.get("bleu", {}).get("finetuned", {}):
            finetuned = results["bleu"]["finetuned"][metric]
        
        # Skip metrics with no data
        if baseline is None and finetuned is None:
            continue
            
        # Create row based on available data
        if has_baseline and has_finetuned and baseline is not None and finetuned is not None:
            improvement = ((finetuned - baseline) / max(baseline, 1e-10)) * 100
            table_data.append([
                metric,
                f"{baseline:.4f}",
                f"{finetuned:.4f}",
                f"{improvement:+.2f}%"
            ])
        elif has_baseline and baseline is not None:
            table_data.append([metric, f"{baseline:.4f}"])
        elif has_finetuned and finetuned is not None:
            table_data.append([metric, f"{finetuned:.4f}"])
    
    if not table_data:
        print("No data available for table visualization")
        return
        
    # Create table
    tab = ax.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    tab.scale(1.2, 1.5)
    
    # Color positive improvements green, negative red if we have improvement column
    if has_baseline and has_finetuned:
        for i, row in enumerate(table_data):
            if len(row) > 3:  # Make sure we have an improvement column
                improvement_text = row[3]
                if improvement_text.startswith('+'):
                    tab[(i+1, 3)].set_facecolor('#d8f3dc')  # Light green
                elif improvement_text.startswith('-'):
                    tab[(i+1, 3)].set_facecolor('#ffccd5')  # Light red
    
    plt.title('Summary Quality Metrics Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(vis_dir / "metrics_table.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate causal LLM summarization quality")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with baseline and fine-tuned summaries")
    parser.add_argument("--output", type=str, help="Output JSON file for evaluation results")
    parser.add_argument("--reference-field", type=str, default="summary", 
                      help="Field name for reference summaries. If not provided, baseline summaries are used as reference")
    parser.add_argument("--baseline-field", type=str, default="llm_summary",
                       help="Field name for baseline causal LLM summaries")
    parser.add_argument("--finetuned-field", type=str, default="finetuned_llm_summary",
                       help="Field name for fine-tuned causal LLM summaries")
    args = parser.parse_args()
    
    # Load articles with summaries
    articles = load_articles(args.input)
    
    # Set default output file if not provided
    if not args.output:
        input_path = Path(args.input)
        output_dir = RESULTS_DIR / f"{input_path.stem}_llm_evaluation"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "results.json"
    else:
        output_file = Path(args.output)
        output_dir = output_file.parent
        output_dir.mkdir(exist_ok=True)
    
    # Extract summaries
    baseline_summaries = []
    finetuned_summaries = []
    reference_summaries = [] if args.reference_field else None
    
    for article in articles:
        has_baseline = args.baseline_field in article
        has_finetuned = args.finetuned_field in article
        
        # Only process articles that have at least one of the required fields
        if not has_baseline and not has_finetuned:
            continue
        
        # Add baseline summary if available
        if has_baseline:
            baseline_summaries.append(article[args.baseline_field])
        
        # Add finetuned summary if available
        if has_finetuned:
            finetuned_summaries.append(article[args.finetuned_field])
        
        # Add reference summary if specified
        if args.reference_field:
            if args.reference_field in article:
                reference_summaries.append(article[args.reference_field])
            else:
                # Use empty string if reference not available
                reference_summaries.append("")
    
    # Report data size
    print(f"Found {len(baseline_summaries)} baseline summaries and {len(finetuned_summaries)} fine-tuned summaries")
    
    # Calculate metrics
    results = {}
    
    print("\nCalculating ROUGE scores...")
    results["rouge"] = compute_rouge_scores(
        baseline_summaries, finetuned_summaries, reference_summaries
    )
    
    print("\nCalculating BERTScore...")
    results["bertscore"] = compute_bertscore(
        baseline_summaries, finetuned_summaries, reference_summaries
    )
    
    print("\nCalculating BLEU scores...")
    results["bleu"] = compute_bleu_scores(
        baseline_summaries, finetuned_summaries, reference_summaries
    )
    
    # Visualize results
    print("\nCreating visualizations...")
    visualize_results(results, output_dir)
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Visualizations saved to {output_dir}/visualizations/")
    
    # Print summary of results
    print("\nSummary of evaluation results:")
    print("=============================")
    
    def print_metric(name, baseline, finetuned):
        if baseline is not None and finetuned is not None:
            improvement = ((finetuned - baseline) / max(baseline, 1e-10)) * 100
            print(f"{name:10s}: Baseline={baseline:.4f}, Fine-tuned={finetuned:.4f} " +
                  f"({improvement:+.2f}% improvement)")
        elif baseline is not None:
            print(f"{name:10s}: Baseline={baseline:.4f}")
        elif finetuned is not None:
            print(f"{name:10s}: Fine-tuned={finetuned:.4f}")
    
    for metric in ["rouge1", "rouge2", "rougeL"]:
        baseline = results["rouge"]["baseline"].get(metric)
        finetuned = results["rouge"]["finetuned"].get(metric)
        print_metric(metric, baseline, finetuned)
    
    baseline = results["bertscore"]["baseline"].get("bertscore")
    finetuned = results["bertscore"]["finetuned"].get("bertscore")
    print_metric("BERTScore", baseline, finetuned)
    
    baseline = results["bleu"]["baseline"].get("bleu")
    finetuned = results["bleu"]["finetuned"].get("bleu")
    print_metric("BLEU", baseline, finetuned)


if __name__ == "__main__":
    main() 