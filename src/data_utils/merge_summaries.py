#!/usr/bin/env python3
"""
Script to merge baseline and finetuned summaries into a single file for metrics evaluation.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of article dictionaries
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} articles from {file_path}")
    return data

def merge_summaries(
    baseline_file: str, 
    finetuned_file: str, 
    output_file: str,
    id_key: str = "id"
) -> None:
    """
    Merge baseline and finetuned summaries into a single file.
    
    Args:
        baseline_file: Path to the file with baseline summaries
        finetuned_file: Path to the file with finetuned summaries
        output_file: Path to save the merged file
        id_key: Key to identify and match articles
    """
    # Load both files
    baseline_articles = load_json_file(baseline_file)
    finetuned_articles = load_json_file(finetuned_file)
    
    # Create dictionaries keyed by article ID for faster lookup
    baseline_dict = {article.get(id_key, i): article for i, article in enumerate(baseline_articles)}
    finetuned_dict = {article.get(id_key, i): article for i, article in enumerate(finetuned_articles)}
    
    # Merge articles
    merged_articles = []
    
    # Process all articles from the finetuned file
    for article_id, finetuned_article in finetuned_dict.items():
        merged_article = finetuned_article.copy()
        
        # If the article exists in the baseline file, add the baseline summary
        if article_id in baseline_dict:
            baseline_article = baseline_dict[article_id]
            
            # Add baseline summary if it exists
            if "summary" in baseline_article:
                merged_article["baseline_summary"] = baseline_article["summary"]
            elif "baseline_summary" in baseline_article:
                merged_article["baseline_summary"] = baseline_article["baseline_summary"]
        
        # Ensure finetuned summary is properly named
        if "summary" in merged_article and "finetuned_summary" not in merged_article:
            merged_article["finetuned_summary"] = merged_article["summary"]
            
        # Add to merged list if it has at least one summary
        if "baseline_summary" in merged_article or "finetuned_summary" in merged_article:
            merged_articles.append(merged_article)
    
    # Add any articles from baseline that weren't in finetuned
    for article_id, baseline_article in baseline_dict.items():
        if article_id not in finetuned_dict:
            merged_article = baseline_article.copy()
            
            # Ensure baseline summary is properly named
            if "summary" in merged_article and "baseline_summary" not in merged_article:
                merged_article["baseline_summary"] = merged_article["summary"]
                
            # Add to merged list if it has at least one summary
            if "baseline_summary" in merged_article:
                merged_articles.append(merged_article)
    
    # Save merged file
    with open(output_file, 'w') as f:
        json.dump(merged_articles, f, indent=2)
    
    print(f"Merged {len(merged_articles)} articles into {output_file}")
    print(f"Articles with baseline summaries: {sum(1 for a in merged_articles if 'baseline_summary' in a)}")
    print(f"Articles with finetuned summaries: {sum(1 for a in merged_articles if 'finetuned_summary' in a)}")
    print(f"Articles with both summaries: {sum(1 for a in merged_articles if 'baseline_summary' in a and 'finetuned_summary' in a)}")

def main():
    parser = argparse.ArgumentParser(description="Merge baseline and finetuned summaries")
    parser.add_argument("--baseline", type=str, required=True, help="Path to the file with baseline summaries")
    parser.add_argument("--finetuned", type=str, required=True, help="Path to the file with finetuned summaries")
    parser.add_argument("--output", type=str, required=True, help="Path to save the merged file")
    parser.add_argument("--id-key", type=str, default="id", help="Key to identify and match articles")
    
    args = parser.parse_args()
    
    merge_summaries(
        baseline_file=args.baseline,
        finetuned_file=args.finetuned,
        output_file=args.output,
        id_key=args.id_key
    )

if __name__ == "__main__":
    main() 