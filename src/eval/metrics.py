#!/usr/bin/env python3
"""
Script to evaluate summary quality using ROUGE, BERTScore, and factuality checks.
Compares baseline and fine-tuned model summaries.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torchtext.data.metrics import bleu_score as torchtext_bleu_score

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


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


def compute_rouge_scores(articles: List[Dict[str, Any]], reference_key: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute ROUGE scores for baseline and fine-tuned summaries.
    
    Args:
        articles: List of article dictionaries with summaries
        reference_key: Key for the reference summary
        
    Returns:
        Tuple of (baseline_scores, finetuned_scores)
    """
    print("Computing ROUGE scores...")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize results
    baseline_scores = {
        "rouge1": [], "rouge2": [], "rougeL": []
    }
    finetuned_scores = {
        "rouge1": [], "rouge2": [], "rougeL": []
    }
    
    for article in tqdm(articles, desc="ROUGE evaluation"):
        reference = article.get(reference_key)
        baseline = article.get("baseline_summary")
        finetuned = article.get("finetuned_summary")
        
        # Skip if any summary is missing
        if not reference or not baseline or not finetuned:
            continue
        
        # Compute baseline scores
        baseline_score = scorer.score(reference, baseline)
        baseline_scores["rouge1"].append(baseline_score["rouge1"].fmeasure)
        baseline_scores["rouge2"].append(baseline_score["rouge2"].fmeasure)
        baseline_scores["rougeL"].append(baseline_score["rougeL"].fmeasure)
        
        # Compute fine-tuned scores
        finetuned_score = scorer.score(reference, finetuned)
        finetuned_scores["rouge1"].append(finetuned_score["rouge1"].fmeasure)
        finetuned_scores["rouge2"].append(finetuned_score["rouge2"].fmeasure)
        finetuned_scores["rougeL"].append(finetuned_score["rougeL"].fmeasure)
    
    # Calculate average scores
    for key in baseline_scores:
        baseline_scores[key] = {
            "scores": baseline_scores[key],
            "mean": np.mean(baseline_scores[key]),
            "std": np.std(baseline_scores[key])
        }
        
        finetuned_scores[key] = {
            "scores": finetuned_scores[key],
            "mean": np.mean(finetuned_scores[key]),
            "std": np.std(finetuned_scores[key])
        }
    
    return baseline_scores, finetuned_scores


def compute_bertscore(articles: List[Dict[str, Any]], reference_key: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute BERTScore for baseline and fine-tuned summaries.
    
    Args:
        articles: List of article dictionaries with summaries
        reference_key: Key for the reference summary
        
    Returns:
        Tuple of (baseline_scores, finetuned_scores)
    """
    print("Computing BERTScore...")
    
    # Extract summaries
    references = []
    baseline_candidates = []
    finetuned_candidates = []

    print(articles)
    
    # Collect valid summary pairs
    for article in articles:
        reference = article.get(reference_key)
        baseline = article.get("baseline_summary")
        finetuned = article.get("finetuned_summary")
        
        # Skip if any summary is missing
        if not reference or not baseline or not finetuned:
            continue
            
        references.append(reference)
        baseline_candidates.append(baseline)
        finetuned_candidates.append(finetuned)

    print(baseline_candidates)
    print(finetuned_candidates)
    print(references)
    
    # Compute BERTScore in a single batch for each model
    P_baseline, R_baseline, F1_baseline = bert_score(baseline_candidates, references, lang="en", verbose=True)
    P_finetuned, R_finetuned, F1_finetuned = bert_score(finetuned_candidates, references, lang="en", verbose=True)
    
    # Create results dictionaries with numpy arrays
    baseline_scores = {
        "precision": P_baseline.numpy(),
        "recall": R_baseline.numpy(),
        "f1": F1_baseline.numpy(),
        "mean": F1_baseline.mean().item(),
        "std": F1_baseline.std().item()
    }
    
    finetuned_scores = {
        "precision": P_finetuned.numpy(),
        "recall": R_finetuned.numpy(),
        "f1": F1_finetuned.numpy(),
        "mean": F1_finetuned.mean().item(),
        "std": F1_finetuned.std().item()
    }
    
    return baseline_scores, finetuned_scores


def compute_bleu_scores(articles: List[Dict[str, Any]], reference_key: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute BLEU scores for baseline and fine-tuned summaries using torchtext.
    
    Args:
        articles: List of article dictionaries with summaries
        reference_key: Key for the reference summary
        
    Returns:
        Tuple of (baseline_scores, finetuned_scores)
    """
    print("Computing BLEU scores...")
    
    # Make sure NLTK has necessary data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Initialize results
    baseline_bleu_scores = []
    finetuned_bleu_scores = []
    
    for article in tqdm(articles, desc="BLEU evaluation"):
        reference = article.get(reference_key)
        baseline = article.get("baseline_summary")
        finetuned = article.get("finetuned_summary")
        
        # Skip if any summary is missing
        if not reference or not baseline or not finetuned:
            continue
        
        # Tokenize summaries
        reference_tokens = nltk.word_tokenize(reference.lower())
        baseline_tokens = nltk.word_tokenize(baseline.lower())
        finetuned_tokens = nltk.word_tokenize(finetuned.lower())
        
        # Skip if any tokenization is empty
        if not reference_tokens or not baseline_tokens or not finetuned_tokens:
            continue
        
        # Calculate BLEU score (torchtext expects a list of token sequences)
        baseline_score = torchtext_bleu_score([baseline_tokens], [[reference_tokens]])
        finetuned_score = torchtext_bleu_score([finetuned_tokens], [[reference_tokens]])
        
        baseline_bleu_scores.append(baseline_score)
        finetuned_bleu_scores.append(finetuned_score)
    
    # Create results dictionaries
    baseline_scores = {
        "bleu": {
            "scores": baseline_bleu_scores,
            "mean": np.mean(baseline_bleu_scores),
            "std": np.std(baseline_bleu_scores)
        }
    }
    
    finetuned_scores = {
        "bleu": {
            "scores": finetuned_bleu_scores,
            "mean": np.mean(finetuned_bleu_scores),
            "std": np.std(finetuned_bleu_scores)
        }
    }
    
    return baseline_scores, finetuned_scores


def check_factuality(articles: List[Dict[str, Any]], device: int = -1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Check factuality of summaries using NLI.
    
    Args:
        articles: List of article dictionaries with summaries
        device: Device to use (-1 for CPU, 0+ for specific GPU)
        
    Returns:
        Tuple of (baseline_results, finetuned_results)
    """
    print("Checking factuality of summaries...")
    
    # Check if CUDA is available and set device accordingly
    if torch.cuda.is_available() and device >= 0:
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        device_str = f"cuda:{device}"
    else:
        print("Using CPU for inference")
        device_str = "cpu"
        device = -1
    
    # Load NLI model
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device_str)
    
    # Create NLI pipeline
    nli = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Initialize results
    baseline_results = {
        "entailment_scores": [],
        "contradiction_scores": [],
        "hallucinations": 0
    }
    
    finetuned_results = {
        "entailment_scores": [],
        "contradiction_scores": [],
        "hallucinations": 0
    }
    
    for article in tqdm(articles, desc="Factuality checking"):
        text = article.get("cleaned_text", article.get("text", ""))
        baseline = article.get("baseline_summary", "")
        finetuned = article.get("finetuned_summary", "")
        
        # Skip if any summary or text is missing
        if not text or not baseline or not finetuned:
            continue
        
        # Check baseline factuality
        try:
            result = nli(baseline, [text], hypothesis_template="This text states: {}")
            entailment_idx = result["labels"].index("ENTAILMENT")
            contradiction_idx = result["labels"].index("CONTRADICTION")
            
            entailment_score = result["scores"][entailment_idx]
            contradiction_score = result["scores"][contradiction_idx]
            
            baseline_results["entailment_scores"].append(entailment_score)
            baseline_results["contradiction_scores"].append(contradiction_score)
            
            # Consider a hallucination if contradiction > entailment
            if contradiction_score > entailment_score:
                baseline_results["hallucinations"] += 1
        except Exception as e:
            print(f"Error checking baseline factuality: {str(e)}")
        
        # Check fine-tuned factuality
        try:
            result = nli(finetuned, [text], hypothesis_template="This text states: {}")
            entailment_idx = result["labels"].index("ENTAILMENT")
            contradiction_idx = result["labels"].index("CONTRADICTION")
            
            entailment_score = result["scores"][entailment_idx]
            contradiction_score = result["scores"][contradiction_idx]
            
            finetuned_results["entailment_scores"].append(entailment_score)
            finetuned_results["contradiction_scores"].append(contradiction_score)
            
            # Consider a hallucination if contradiction > entailment
            if contradiction_score > entailment_score:
                finetuned_results["hallucinations"] += 1
        except Exception as e:
            print(f"Error checking fine-tuned factuality: {str(e)}")
    
    # Calculate averages
    num_articles = len(baseline_results["entailment_scores"])
    baseline_results["avg_entailment"] = np.mean(baseline_results["entailment_scores"])
    baseline_results["avg_contradiction"] = np.mean(baseline_results["contradiction_scores"])
    baseline_results["hallucination_rate"] = baseline_results["hallucinations"] / num_articles if num_articles > 0 else 0
    
    finetuned_results["avg_entailment"] = np.mean(finetuned_results["entailment_scores"])
    finetuned_results["avg_contradiction"] = np.mean(finetuned_results["contradiction_scores"])
    finetuned_results["hallucination_rate"] = finetuned_results["hallucinations"] / num_articles if num_articles > 0 else 0
    
    return baseline_results, finetuned_results


def compute_compression_ratio(articles: List[Dict[str, Any]], reference_key: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute compression ratio (article word count / summary word count) for baseline and fine-tuned summaries.
    
    Args:
        articles: List of article dictionaries with summaries
        reference_key: Key for the reference summary
        
    Returns:
        Tuple of (baseline_scores, finetuned_scores)
    """
    print("Computing compression ratios...")
    
    # Initialize results
    baseline_compression = []
    finetuned_compression = []
    
    for article in tqdm(articles, desc="Compression evaluation"):

        article_text_str = "cleaned_text"

        article_text = article.get(article_text_str)
        # Check if article text exists
        
        if not article_text:
            continue
            
        baseline = article.get("baseline_summary")
        finetuned = article.get("finetuned_summary")
        
        # Skip if any summary is missing
        if not baseline or not finetuned:
            continue
        
        # Count words
        article_word_count = len(article_text.split())
        baseline_word_count = len(baseline.split())
        finetuned_word_count = len(finetuned.split())
        
        # Calculate compression ratio (article words / summary words)
        if baseline_word_count > 0:
            baseline_compression.append(article_word_count / baseline_word_count)
        
        if finetuned_word_count > 0:
            finetuned_compression.append(article_word_count / finetuned_word_count)
    
    # Calculate average scores
    baseline_scores = {
        "scores": baseline_compression,
        "mean": np.mean(baseline_compression),
        "std": np.std(baseline_compression)
    }
    
    finetuned_scores = {
        "scores": finetuned_compression,
        "mean": np.mean(finetuned_compression),
        "std": np.std(finetuned_compression)
    }
    
    return baseline_scores, finetuned_scores


def generate_plots(
    rouge_baseline,
    rouge_finetuned,
    bertscore_baseline,
    bertscore_finetuned,
    bleu_baseline,
    bleu_finetuned,
    compression_baseline,
    compression_finetuned,
    # factuality_baseline,
    # factuality_finetuned,
    output_prefix
):
    """
    Generate plots to visualize evaluation results.
    
    Args:
        rouge_baseline: ROUGE scores for baseline model
        rouge_finetuned: ROUGE scores for fine-tuned model
        bertscore_baseline: BERTScore for baseline model
        bertscore_finetuned: BERTScore for fine-tuned model
        bleu_baseline: BLEU scores for baseline model
        bleu_finetuned: BLEU scores for fine-tuned model
        compression_baseline: Compression ratio for baseline model
        compression_finetuned: Compression ratio for fine-tuned model
        factuality_baseline: Factuality scores for baseline model
        factuality_finetuned: Factuality scores for fine-tuned model
        output_prefix: Prefix for output files
    """
    print("Generating evaluation plots...")
    
    # Set up styling
    plt.style.use('ggplot')
    
    # 1. ROUGE Scores Comparison
    plt.figure(figsize=(10, 6))
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    baseline_means = [
        rouge_baseline['rouge1']['mean'],
        rouge_baseline['rouge2']['mean'],
        rouge_baseline['rougeL']['mean']
    ]
    finetuned_means = [
        rouge_finetuned['rouge1']['mean'],
        rouge_finetuned['rouge2']['mean'],
        rouge_finetuned['rougeL']['mean']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, baseline_means, width, label='Baseline')
    ax.bar(x + width/2, finetuned_means, width, label='Fine-tuned')
    
    ax.set_ylabel('Score')
    ax.set_title('ROUGE Scores Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_rouge.png")
    
    # 2. BLEU Scores Comparison
    plt.figure(figsize=(8, 6))
    labels = ['Baseline', 'Fine-tuned']
    bleu_means = [bleu_baseline['bleu']['mean'], bleu_finetuned['bleu']['mean']]
    
    plt.bar(labels, bleu_means)
    plt.ylabel('Score')
    plt.title('BLEU Score Comparison')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bleu.png")
    
    # 3. BERTScore Comparison
    plt.figure(figsize=(10, 6))
    metrics = ['Precision', 'Recall', 'F1']
    baseline_means = [
        bertscore_baseline['mean'],
        bertscore_baseline['recall'].mean(),
        bertscore_baseline['f1'].mean()
    ]
    finetuned_means = [
        bertscore_finetuned['mean'],
        bertscore_finetuned['recall'].mean(),
        bertscore_finetuned['f1'].mean()
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, baseline_means, width, label='Baseline')
    ax.bar(x + width/2, finetuned_means, width, label='Fine-tuned')
    
    ax.set_ylabel('Score')
    ax.set_title('BERTScore Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bertscore.png")
    
    # 4. Compression Ratio Comparison
    plt.figure(figsize=(10, 6))
    plt.hist(compression_baseline["scores"], alpha=0.5, label='Baseline', bins=30)
    plt.hist(compression_finetuned["scores"], alpha=0.5, label='Fine-tuned', bins=30)
    plt.axvline(compression_baseline["mean"], color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(compression_finetuned["mean"], color='orange', linestyle='dashed', linewidth=1)
    plt.title(f'Compression Ratio Distribution (Article words / Summary words)')
    plt.xlabel('Compression Ratio')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_compression.png")
    plt.close()
    
    # 5. Combined Metrics Comparison (ROUGE-1, BLEU-Cumulative, BERTScore)
    plt.figure(figsize=(12, 8))
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'BLEU'] # 'Compression Ratio'
    baseline_means = [
        rouge_baseline["rouge1"]["mean"] * 100, 
        rouge_baseline["rouge2"]["mean"] * 100, 
        rouge_baseline["rougeL"]["mean"] * 100,
        bertscore_baseline["mean"] * 100,
        bleu_baseline["bleu"]["mean"] *100,
        #compression_baseline["mean"] / 10  # Scale down for visualization
    ]
    finetuned_means = [
        rouge_finetuned["rouge1"]["mean"] * 100, 
        rouge_finetuned["rouge2"]["mean"] * 100, 
        rouge_finetuned["rougeL"]["mean"] * 100,
        bertscore_finetuned["mean"] * 100,
        bleu_finetuned["bleu"]["mean"] * 100,
        #compression_finetuned["mean"] / 10  # Scale down for visualization
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, baseline_means, width, label='Baseline')
    rects2 = ax.bar(x + width/2, finetuned_means, width, label='Fine-tuned')
    
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    # # Add a note about compression ratio scaling
    # plt.figtext(0.99, 0.01, "* Compression Ratio is scaled down by factor of 10 for visualization", 
    #             wrap=True, horizontalalignment='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_combined.png")
    
    print(f"Plots saved with prefix: {output_prefix}")


def generate_report(
    articles,
    rouge_baseline,
    rouge_finetuned,
    bertscore_baseline,
    bertscore_finetuned,
    bleu_baseline,
    bleu_finetuned,
    compression_baseline,
    compression_finetuned,
    # factuality_baseline,
    # factuality_finetuned,
    output_file
):
    """
    Generate evaluation report.
    
    Args:
        articles: List of article dictionaries with summaries
        rouge_baseline: ROUGE scores for baseline model
        rouge_finetuned: ROUGE scores for fine-tuned model
        bertscore_baseline: BERTScore for baseline model
        bertscore_finetuned: BERTScore for fine-tuned model
        bleu_baseline: BLEU scores for baseline model
        bleu_finetuned: BLEU scores for fine-tuned model
        compression_baseline: Compression ratio for baseline model
        compression_finetuned: Compression ratio for fine-tuned model
        factuality_baseline: Factuality scores for baseline model
        factuality_finetuned: Factuality scores for fine-tuned model
        output_file: Output file path
    """
    print("Generating evaluation report...")
    
    report = {
        "summary": {
            "num_articles": len(articles),
            "baseline": {
                "rouge1": rouge_baseline["rouge1"]["mean"],
                "rouge2": rouge_baseline["rouge2"]["mean"],
                "rougeL": rouge_baseline["rougeL"]["mean"],
                "bertscore": bertscore_baseline["mean"],
                "bleu": bleu_baseline["bleu"]["mean"],
                # "entailment": factuality_baseline["avg_entailment"],
                # "contradiction": factuality_baseline["avg_contradiction"],
                # "hallucination_rate": factuality_baseline["hallucination_rate"]
            },
            "finetuned": {
                "rouge1": rouge_finetuned["rouge1"]["mean"],
                "rouge2": rouge_finetuned["rouge2"]["mean"],
                "rougeL": rouge_finetuned["rougeL"]["mean"],
                "bertscore": bertscore_finetuned["mean"],
                "bleu": bleu_finetuned["bleu"]["mean"],
                    # "entailment": factuality_finetuned["avg_entailment"],
                    # "contradiction": factuality_finetuned["avg_contradiction"],
                    # "hallucination_rate": factuality_finetuned["hallucination_rate"]
            }
        },
        "articles": articles
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Report saved to {output_file}")
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Number of articles: {len(articles)}")
    print("\nROUGE Scores:")
    print(f"  ROUGE-1: Baseline = {rouge_baseline['rouge1']['mean']:.4f}, Fine-tuned = {rouge_finetuned['rouge1']['mean']:.4f}")
    print(f"  ROUGE-2: Baseline = {rouge_baseline['rouge2']['mean']:.4f}, Fine-tuned = {rouge_finetuned['rouge2']['mean']:.4f}")
    print(f"  ROUGE-L: Baseline = {rouge_baseline['rougeL']['mean']:.4f}, Fine-tuned = {rouge_finetuned['rougeL']['mean']:.4f}")
    
    print("\nBLEU Scores:")
    print(f"  BLEU: Baseline = {bleu_baseline['bleu']['mean']:.4f}, Fine-tuned = {bleu_finetuned['bleu']['mean']:.4f}")
    
    print("\nBERTScore:")
    print(f"  Baseline = {bertscore_baseline['mean']:.4f}, Fine-tuned = {bertscore_finetuned['mean']:.4f}")
    
    print("\nCompression Ratio:")
    print(f"  Baseline = {compression_baseline['mean']:.4f}, Fine-tuned = {compression_finetuned['mean']:.4f}")
    
    print("\nFactuality:")
    # print(f"  Entailment: Baseline = {factuality_baseline['avg_entailment']:.4f}, Fine-tuned = {factuality_finetuned['avg_entailment']:.4f}")
    # print(f"  Contradiction: Baseline = {factuality_baseline['avg_contradiction']:.4f}, Fine-tuned = {factuality_finetuned['avg_contradiction']:.4f}")
    # print(f"  Hallucination Rate: Baseline = {factuality_baseline['hallucination_rate']:.2f}, Fine-tuned = {factuality_finetuned['hallucination_rate']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate summary quality and compare models")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file containing articles with summaries")
    parser.add_argument("--output", type=str, required=False, default="output", help="Output JSON file for evaluation report")
    parser.add_argument("--reference", type=str, default="summary",
                        help="Field name for reference summaries (gpt_summary or summary)")
    parser.add_argument("--device", type=int, default=-1,
                        help="Device to use for factuality checks (-1 for CPU, 0+ for specific GPU)")
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(DATA_DIR / f"{input_path.stem}_evaluation.json")
    
    # Load articles
    articles = load_articles(args.input)
    
    # Filter articles that have both baseline and fine-tuned summaries
    articles = [
        article for article in articles
        if article.get("baseline_summary") and article.get("finetuned_summary") and article.get(args.reference)
    ]
    
    print(f"Evaluating {len(articles)} articles with both summaries and references")
    
    # Compute ROUGE scores
    rouge_baseline, rouge_finetuned = compute_rouge_scores(articles, args.reference)
    
    # Compute BERTScore
    bertscore_baseline, bertscore_finetuned = compute_bertscore(articles, args.reference)
    
    # Compute BLEU scores
    bleu_baseline, bleu_finetuned = compute_bleu_scores(articles, args.reference)
    
    # Compute compression ratio
    compression_baseline, compression_finetuned = compute_compression_ratio(articles, args.reference)
    
    # # Check factuality
    # factuality_baseline, factuality_finetuned = check_factuality(articles, args.device)
    
    # Generate plots
    output_prefix = str(Path(args.output).with_suffix(''))
    generate_plots(
        rouge_baseline,
        rouge_finetuned,
        bertscore_baseline,
        bertscore_finetuned,
        bleu_baseline,
        bleu_finetuned,
        compression_baseline,
        compression_finetuned,
        # factuality_baseline,
        # factuality_finetuned,
        output_prefix
    )
    
    # Generate report
    generate_report(
        articles,
        rouge_baseline,
        rouge_finetuned,
        bertscore_baseline,
        bertscore_finetuned,
        bleu_baseline,
        bleu_finetuned,
        compression_baseline,
        compression_finetuned,
        # factuality_baseline,
        # factuality_finetuned,
        args.output
    )


if __name__ == "__main__":
    main()