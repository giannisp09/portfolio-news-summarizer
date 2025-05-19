#!/usr/bin/env python3
"""
Script to evaluate LLM summarization performance using GPT-4 as judge.
Compares baseline instruction-tuned LLMs with fine-tuned versions.
"""

from openai import OpenAI
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Data directories
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Evaluation prompt template based on G-Eval
EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.
Answer only with a number between 1 and 5 hence the score

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{document}

Summary:

{summary}

Evaluation Form (scores ONLY):

- {metric_name}
"""

# Metric 1: Relevance

RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source. \
The summary should include only important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""

# Metric 2: Coherence

COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic."
"""

COHERENCE_SCORE_STEPS = """
1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 3: Consistency

CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.
"""

# Metric 4: Fluency

FLUENCY_SCORE_CRITERIA = """
Fluency(1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.
"""

# Metric 5: Hallucination

HALLUCINATION_SCORE_CRITERIA = """
Hallucination(1-5) - the presence of false information in the summary. \
Hallucination is the presence of information that is not present in the source document.
"""

HALLUCINATION_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Identify any information in the summary that is not present in the source document. Note information should be something complete different from the source document and not just a different way of saying the same thing.
3. Assign a hallucination score from 1 to 5. the more false information, the higher the score.
"""

# Metric 6: Factual Accuracy

FACTUAL_ACCURACY_SCORE_CRITERIA = """
Factual Accuracy(1-5) - the accuracy of the summary in terms of the facts and details it presents.
"""

FACTUAL_ACCURACY_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Identify any information in the summary that is not present in the source document. Note information should be something complete different from the source document and not just a different way of saying the same thing.
so if sumamry contains paraphrased information from the source document, it is still factual if it captures the main idea of the source document.
3. Assign a factual accuracy score from 1 to 5. the more false information, the higher the score.
"""

# Metric 7: Conciseness

CONCISENESS_SCORE_CRITERIA = """
Conciseness(1-5) - the length of the summary.
"""

CONCISENESS_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Assess the length of the summary and compare it to the source document.
3. Assign a conciseness score from 1 to 5. the shorter the summary, the higher the score.
"""





def get_geval_score(
    criteria: str, steps: str, document: str, summary: str, metric_name: str
):
    """
    Get evaluation score from GPT-4 based on criteria and steps.
    
    Args:
        criteria: Evaluation criteria description
        steps: Evaluation steps to follow
        document: Source document text
        summary: Summary to evaluate
        metric_name: Name of the metric being evaluated
        
    Returns:
        Score as an integer
    """
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        document=document,
        summary=summary,
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        result = response.choices[0].message.content.strip()
        
        # Extract numeric score using regex
        match = re.search(r'(\d+)', result)
        
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: Could not extract score from response: {result}")
            return None
    except Exception as e:
        print(f"Error getting evaluation: {e}")
        return None


evaluation_metrics = {
    "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
    "Hallucination": (HALLUCINATION_SCORE_CRITERIA, HALLUCINATION_SCORE_STEPS),
    "Factuality": (FACTUAL_ACCURACY_SCORE_CRITERIA, FACTUAL_ACCURACY_SCORE_STEPS),
    "Conciseness": (CONCISENESS_SCORE_CRITERIA, CONCISENESS_SCORE_STEPS),
}


def load_articles(input_file: str):
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
    return articles


def visualize_results(results_df, output_dir):
    """
    Create visualizations for the evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save visualizations
    """
    # Create a directory for visualizations
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Calculate mean scores by summary type and metric
    pivot_df = results_df.pivot_table(
        index='Evaluation Type', 
        columns='Summary Type', 
        values='Score',
        aggfunc='mean'
    )
    
    # Bar chart comparing metrics
    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind='bar', figsize=(12, 6))
    plt.title('LLM Judge Evaluation Scores by Metric')
    plt.xlabel('Evaluation Metric')
    plt.ylabel('Average Score')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Summary Type')
    plt.tight_layout()
    plt.savefig(vis_dir / "evaluation_by_metric.png", dpi=300)
    plt.close()
    
    # Stacked bar chart
    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('LLM Judge Evaluation - Stacked Comparison')
    plt.xlabel('Evaluation Metric')
    plt.ylabel('Score')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Summary Type')
    plt.tight_layout()
    plt.savefig(vis_dir / "evaluation_stacked.png", dpi=300)
    plt.close()

    # Grouped bar chart by summary type
    pivot_df_by_summary = results_df.pivot_table(
        index='Summary Type', 
        columns='Evaluation Type', 
        values='Score',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    pivot_df_by_summary.plot(kind='bar', figsize=(12, 6))
    plt.title('LLM Judge Evaluation Scores by Summary Type')
    plt.xlabel('Summary Type')
    plt.ylabel('Average Score')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Evaluation Metric')
    plt.tight_layout()
    plt.savefig(vis_dir / "evaluation_by_summary_type.png", dpi=300)
    plt.close()
    
    # Radar chart if we have multiple metrics and summary types
    if len(pivot_df.index) >= 3 and len(pivot_df.columns) >= 2:
        # Create radar chart
        categories = pivot_df.index
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of categories
        N = len(categories)
        
        # Set angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw one line per summary type and fill area
        for i, (summary_type, values) in enumerate(pivot_df.items()):
            values = values.values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=summary_type)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Set y-axis limits
        if 'Fluency' in pivot_df.index:
            plt.ylim(0, 5)  # All metrics except Fluency go from 1-5
        else:
            plt.ylim(0, 5)  # Default range
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('LLM Judge Evaluation - Radar Chart')
        plt.tight_layout()
        plt.savefig(vis_dir / "evaluation_radar.png", dpi=300)
        plt.close()
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
    plt.title('LLM Judge Evaluation Scores - Heatmap')
    plt.tight_layout()
    plt.savefig(vis_dir / "evaluation_heatmap.png", dpi=300)
    plt.close()
    
    # If we have both baseline and fine-tuned, create improvement comparison
    if 'Baseline LLM' in pivot_df.columns and 'Fine-tuned LLM' in pivot_df.columns:
        # Calculate improvement percentages
        improvement_df = ((pivot_df['Fine-tuned LLM'] - pivot_df['Baseline LLM']) / 
                         pivot_df['Baseline LLM'] * 100)
        
        plt.figure(figsize=(10, 6))
        improvement_df.plot(kind='bar', color='teal')
        plt.title('Percentage Improvement: Fine-tuned vs Baseline')
        plt.xlabel('Evaluation Metric')
        plt.ylabel('Improvement (%)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add percentage labels above/below each bar
        for i, value in enumerate(improvement_df):
            plt.text(i, value + (5 if value >= 0 else -5), 
                    f"{value:.1f}%", 
                    ha='center', va='center' if value >= 0 else 'top',
                    fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(vis_dir / "improvement_percentage.png", dpi=300)
        plt.close()
    
    # Create a summary table image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Add improvement column if both types are present
    if 'Baseline LLM' in pivot_df.columns and 'Fine-tuned LLM' in pivot_df.columns:
        # Calculate improvement
        pivot_df['Improvement'] = ((pivot_df['Fine-tuned LLM'] - pivot_df['Baseline LLM']) / 
                                 pivot_df['Baseline LLM'] * 100)
        pivot_df['Improvement'] = pivot_df['Improvement'].apply(lambda x: f"{x:+.1f}%")
    
    # Create table
    table = ax.table(cellText=pivot_df.reset_index().values,
                    colLabels=['Metric'] + list(pivot_df.columns),
                    loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Customize table appearance
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif j == 0:  # First column
            cell.set_facecolor('#D9E1F2')
        elif j == len(pivot_df.columns) and 'Improvement' in pivot_df.columns:  # Improvement column
            value = cell.get_text().get_text()
            if value.startswith('+'):
                cell.set_facecolor('#d8f3dc')  # Light green for positive
            elif value.startswith('-'):
                cell.set_facecolor('#ffccd5')  # Light red for negative
            
    plt.title('LLM Judge Evaluation Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(vis_dir / "evaluation_table.png", dpi=300)
    plt.close()
    
    return pivot_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM summarization quality using GPT-4 as judge")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with summaries")
    parser.add_argument("--output", type=str, help="Output JSON file for evaluation results")
    parser.add_argument("--content-field", type=str, default="text", 
                      help="Field name for article content")
    parser.add_argument("--baseline-field", type=str, default="baseline_summary",
                       help="Field name for baseline LLM summaries")
    parser.add_argument("--finetuned-field", type=str, default="finetuned_summary",
                       help="Field name for fine-tuned LLM summaries")
    parser.add_argument("--reference-field", type=str, default="summary",
                       help="Field name for reference summaries (optional)")
    parser.add_argument("--sample", type=int, default=None,
                      help="Number of articles to sample for evaluation (to save API costs)")
    parser.add_argument("--metrics", type=str, nargs='+', 
                       choices=['Relevance', 'Coherence', 'Consistency', 'Fluency', 'Hallucination', 'Factuality', 'Conciseness', 'all'],
                       default=['all'],
                       help="Metrics to evaluate (default: all)")
    args = parser.parse_args()
    
    # Load articles with summaries
    articles = load_articles(args.input)
    
    # Sample articles if requested
    if args.sample and args.sample < len(articles):
        import random
        articles = random.sample(articles, args.sample)
        print(f"Sampled {args.sample} articles for evaluation")
    
    # Determine which metrics to evaluate
    metrics_to_evaluate = list(evaluation_metrics.keys()) if 'all' in args.metrics else args.metrics
    
    # Set default output file if not provided
    if not args.output:
        input_path = Path(args.input)
        output_dir = RESULTS_DIR / f"{input_path.stem}_llm_judge_evaluation"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "results.json"
    else:
        output_file = Path(args.output)
        output_dir = output_file.parent
        output_dir.mkdir(exist_ok=True)
    
    # Initialize results data structure
    data = {"Evaluation Type": [], "Summary Type": [], "Score": [], "Article ID": []}
    
    # Loop through articles
    for i, article in enumerate(tqdm(articles, desc="Evaluating articles")):
        article_id = article.get("id", i)
        
        # Get article content
        if args.content_field not in article:
            print(f"Warning: Article {article_id} missing content field. Skipping.")
            continue
            
        document = article[args.content_field]
        
        # Check which summaries are available
        has_baseline = args.baseline_field in article
        has_finetuned = args.finetuned_field in article
        has_reference = args.reference_field in article
        
        # Skip if no summaries to evaluate
        if not has_baseline and not has_finetuned and not has_reference:
            print(f"Warning: Article {article_id} has no summaries to evaluate. Skipping.")
            continue
        
        # Define which summaries to evaluate
        summaries_to_evaluate = {}
        if has_baseline:
            summaries_to_evaluate["Baseline LLM"] = article[args.baseline_field]
        if has_finetuned:
            summaries_to_evaluate["Fine-tuned LLM"] = article[args.finetuned_field]
        if has_reference:
            summaries_to_evaluate["Reference"] = article[args.reference_field]
        
        # Evaluate each summary on each metric
        for eval_type, (criteria, steps) in evaluation_metrics.items():
            if eval_type not in metrics_to_evaluate:
                continue
                
            for summ_type, summary in summaries_to_evaluate.items():
                # Skip empty summaries
                if not summary:
                    continue

                print(f"Evaluating {eval_type} for {summ_type}")
                    
                score = get_geval_score(criteria, steps, document, summary, eval_type)
                
                if score is not None:
                    data["Evaluation Type"].append(eval_type)
                    data["Summary Type"].append(summ_type)
                    data["Score"].append(score)
                    data["Article ID"].append(article_id)
                    print(f"Score: {score} for {eval_type} for {summ_type}")
                    
    
    # Convert to DataFrame
    results_df = pd.DataFrame(data)
    
    # Create pivot table for easier analysis
    if not results_df.empty:
        pivot_df = visualize_results(results_df, output_dir)
        
        # Save raw results to CSV
        results_df.to_csv(output_dir / "raw_results.csv", index=False)
        
        # Save pivot table to CSV
        pivot_df.to_csv(output_dir / "pivot_results.csv")
        
        # Save results to JSON file
        with open(output_file, 'w') as f:
            json.dump(
                {
                    "raw_results": results_df.to_dict(orient="records"),
                    "summary": pivot_df.reset_index().to_dict(orient="records")
                }, 
                f, 
                indent=2
            )
        
        print(f"\nResults saved to {output_file}")
        print(f"Visualizations saved to {output_dir}/visualizations/")
        
        # Print summary of results
        print("\nSummary of evaluation results:")
        print("=============================")
        print(pivot_df)
        
        # If we have both baseline and fine-tuned, print improvement
        if 'Baseline LLM' in pivot_df.columns and 'Fine-tuned LLM' in pivot_df.columns:
            improvement = ((pivot_df['Fine-tuned LLM'] - pivot_df['Baseline LLM']) / 
                          pivot_df['Baseline LLM'] * 100)
            
            print("\nImprovement Percentages:")
            print("=======================")
            for metric, value in improvement.items():
                print(f"{metric:12s}: {value:+.2f}%")
    else:
        print("No results collected. Check that the input file contains the specified fields.")


if __name__ == "__main__":
    main()