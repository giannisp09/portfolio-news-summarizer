#!/usr/bin/env python3
"""
Streamlit app for interacting with the news summarization system.
Allows users to search for articles by ticker, view summaries, and compare models.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import uuid
import time
import io

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from openpyxl import Workbook
import torch

# Add parent directory to path to import project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import project modules
from src.scraping.scraper import get_yahoo_finance_news, extract_full_text
from src.data_utils.cleaner import clean_text
from src.inference.inference import setup_baseline_model, setup_finetuned_model


# Constants
DEFAULT_BASELINE_MODEL = "facebook/bart-large-cnn"
DEFAULT_FINETUNED_PATH = "models/finetuned/yfinance-md"
DATA_DIR = Path(parent_dir) / "data"


def load_articles() -> List[Dict[str, Any]]:
    """
    Load all available articles from the data directory.
    
    Returns:
        List of article dictionaries
    """
    all_articles = []
    #data_files = list(DATA_DIR.glob('*_comparison.json')) + list(DATA_DIR.glob('*_articles_*.json'))
    data_files = list(DATA_DIR.glob('*eval_yfinance.json'))
    for file_path in data_files:
        try:
            with open(file_path, 'r') as f:
                articles = json.load(f)
                
                # Handle both list and dict formats
                if isinstance(articles, dict) and 'articles' in articles:
                    articles = articles['articles']
                    
                all_articles.extend(articles)
        except Exception as e:
            st.warning(f"Error loading {file_path}: {e}")
    
    return all_articles


def load_evaluation_data() -> Optional[Dict[str, Any]]:
    """
    Load evaluation data if available.
    
    Returns:
        Evaluation data dictionary or None
    """
    eval_files = list(DATA_DIR.glob('evaluation.json'))
    
    if eval_files:
        try:
            with open(eval_files[0], 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading evaluation data: {e}")
    
    return None


def get_available_models() -> Dict[str, List[str]]:
    """
    Scan the models directory for available models.
    
    Returns:
        Dictionary with model types as keys and lists of model paths as values
    """
    models_dir = Path(parent_dir) / "models"
    models = {
        "Baseline": [DEFAULT_BASELINE_MODEL],
        "Fine-tuned": []
    }
    
    # Add standard HuggingFace models
    common_models = [
        "facebook/bart-large-cnn",
        "google/pegasus-xsum",
        "t5-small",
        "t5-base",
        "sshleifer/distilbart-cnn-12-6"
    ]
    
    # Add them to the baseline models
    models["Baseline"].extend(common_models)
    
    # Check if models directory exists
    if models_dir.exists():
        # Find all potential model directories
        finetuned_dirs = list(models_dir.glob("**/config.json"))
        for config_file in finetuned_dirs:
            model_dir = config_file.parent
            # Add relative path to make it easier to read
            rel_path = str(model_dir.relative_to(parent_dir))
            models["Fine-tuned"].append(rel_path)
    
    # Remove duplicates and sort
    models["Baseline"] = sorted(list(set(models["Baseline"])))
    models["Fine-tuned"] = sorted(list(set(models["Fine-tuned"])))
    
    return models


def filter_articles_by_ticker(articles: List[Dict[str, Any]], ticker: str) -> List[Dict[str, Any]]:
    """
    Filter articles by ticker symbol.
    
    Args:
        articles: List of article dictionaries
        ticker: Ticker symbol to filter by
        
    Returns:
        Filtered list of article dictionaries
    """
    ticker = ticker.upper()
    return [article for article in articles if article.get('ticker') == ticker]


def display_article(article: Dict[str, Any], index: int, selected_model: str = "Baseline Model") -> None:
    """
    Display an article and its selected summary.
    
    Args:
        article: Article dictionary
        index: Unique index for this article in the current display context
        selected_model: Selected model to display summaries from
    """
    # Article container with light background and padding
    with st.container():
        # Use a card-like style with padding and border
        st.markdown("""
        <style>
        .article-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border-left: 4px solid #4285F4;
        }
        .summary-box {
            background-color: white;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #eee;
            color: #333;
            font-size: 15px;
            line-height: 1.5;
        }
        .reference-box {
            background-color: #E8F0FE;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #d0e0fc;
            color: #333;
            font-size: 15px;
            line-height: 1.5;
        }
        .keyword-tag {
            background-color: #f0f2f6;
            padding: 3px 8px;
            border-radius: 50px;
            margin-right: 6px;
            font-size: 12px;
            color: #333;
            display: inline-block;
            margin-bottom: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="article-card">', unsafe_allow_html=True)
        
        # Title with slightly larger font - fixing the color from #ff to #333
        st.markdown(f"<h3 style='margin-top:0; color:#fff;'>{article.get('title', 'No Title')}</h3>", unsafe_allow_html=True)
        
        # Metadata row
        col1, col2 = st.columns([3, 1])
        with col1:
            # Source and date with cleaner formatting
            st.markdown(f"<span style='color:#555; font-size:14px;'><b>{article.get('source', 'Unknown')}</b> • {article.get('date', 'Unknown')}</span>", unsafe_allow_html=True)
        
        with col2:
            # Article link aligned right
            if article.get('url'):
                st.markdown(f"<div style='text-align:right'><a href='{article['url']}' target='_blank' style='color:#1E88E5;'>Original Article ↗</a></div>", unsafe_allow_html=True)
        
        # Category badge
        category = article.get('category', 'Other')
        category_colors = {
            "Earnings": "#2E7D32",  # Green
            "Product": "#1565C0",   # Blue
            "M&A": "#6A1B9A",       # Purple
            "Leadership": "#FF6F00", # Amber
            "Legal": "#C62828",     # Red
            "Market": "#00838F",    # Teal
            "Economy": "#4E342E",   # Brown
            "Other": "#616161"      # Grey
        }
        color = category_colors.get(category, "#616161")
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 4px 10px; 
        border-radius: 50px; display: inline-block; font-size: 12px; margin: 10px 0;">
            {category}
        </div>
        """, unsafe_allow_html=True)
        
        # Divider
        st.markdown("<hr style='margin: 12px 0; border: none; height: 1px; background-color: #eee;'>", unsafe_allow_html=True)
        
        # Add model selection dropdown for custom inference
        custom_inference = st.expander("Run Custom Model Inference", expanded=False)
        with custom_inference:
            # Get available models
            available_models = get_available_models()
            
            # Model type selection
            model_type = st.selectbox(
                "Model Type:", 
                list(available_models.keys()),
                key=f"model_type_{index}"
            )
            
            # Model selection based on type
            if model_type in available_models and available_models[model_type]:
                model_path = st.selectbox(
                    f"Select {model_type} Model:", 
                    available_models[model_type],
                    key=f"model_path_{index}"
                )
                
                # Button to run inference
                if st.button("Generate Summary", key=f"generate_btn_{index}"):
                    with st.spinner(f"Generating summary with {model_path}..."):
                        try:
                            # Get the text to summarize
                            text = article.get("cleaned_text", article.get("text", ""))
                            
                            if text:
                                # Set up the appropriate model
                                if model_type == "Baseline":
                                    model = setup_baseline_model(model_path)
                                else:  # Fine-tuned
                                    model = setup_finetuned_model(model_path)
                                
                                # Generate summary
                                start_time = time.time()
                                result = model(
                                    text,
                                    max_length=150,
                                    min_length=40,
                                    do_sample=False
                                )
                                generation_time = time.time() - start_time
                                
                                # Store the result temporarily in session state
                                key = f"custom_summary_{index}_{model_path.replace('/', '_')}"
                                st.session_state[key] = {
                                    "summary": result[0]["summary_text"],
                                    "time": generation_time,
                                    "model": model_path
                                }
                                
                                st.success("Summary generated successfully!")
                            else:
                                st.error("No text available to summarize.")
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
            else:
                st.warning(f"No {model_type} models available.")
            
            # Display any custom summaries from session state
            for key in list(st.session_state.keys()):
                if key.startswith(f"custom_summary_{index}_"):
                    custom_result = st.session_state[key]
                    st.markdown(f"<h4 style='color:#333; margin-top:15px;'>Summary ({custom_result['model']})</h4>", unsafe_allow_html=True)
                    summary_html = custom_result["summary"].replace('\n', '<br>')
                    st.markdown(f"<div class='summary-box'>{summary_html}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align:right; color:#888; font-size:12px;'>Generated in {custom_result['time']:.2f} seconds</div>", unsafe_allow_html=True)
        
        # Display the selected summary with a title matching the model
        if selected_model == "Baseline Model":
            st.markdown("<h4 style='color:#fff;'>Summary (Baseline Model)</h4>", unsafe_allow_html=True)
            summary = article.get('baseline_summary')
            if summary:
                # Convert newlines to <br> tags for proper HTML display
                summary_html = summary.replace('\n', '<br>')
                st.markdown(f"<div class='summary-box'>{summary_html}</div>", unsafe_allow_html=True)
                # Show generation time if available
                if 'baseline_time' in article:
                    st.markdown(f"<div style='text-align:right; color:#888; font-size:12px;'>Generated in {article.get('baseline_time', 0):.2f} seconds</div>", unsafe_allow_html=True)
            else:
                st.info("No baseline summary available for this article.")
        
        elif selected_model == "Fine-tuned Model":
            st.markdown("<h4 style='color:#ffff;'>Summary (Fine-tuned Model)</h4>", unsafe_allow_html=True)
            summary = article.get('finetuned_summary')
            if summary:
                # Convert newlines to <br> tags for proper HTML display
                summary_html = summary.replace('\n', '<br>')
                st.markdown(f"<div class='summary-box'>{summary_html}</div>", unsafe_allow_html=True)
                # Show generation time if available
                if 'finetuned_time' in article:
                    st.markdown(f"<div style='text-align:right; color:#888; font-size:12px;'>Generated in {article.get('finetuned_time', 0):.2f} seconds</div>", unsafe_allow_html=True)
            else:
                st.info("No fine-tuned summary available for this article.")
        
        elif selected_model == "Labeled/Ground Truth":
            st.markdown("<h4 style='color:#fff;'>Reference Summary</h4>", unsafe_allow_html=True)
            summary = article.get('labeled_summary') or article.get('reference_summary') or article.get('ground_truth')
            if summary:
                # Convert newlines to <br> tags for proper HTML display
                summary_html = summary.replace('\n', '<br>')
                st.markdown(f"<div class='reference-box'>{summary_html}</div>", unsafe_allow_html=True)
            else:
                st.info("No reference summary available for this article.")
        
        elif selected_model == "Generate New Summary":
            # This option will be handled separately in the main function
            st.markdown("<h4 style='color:#fff;'>Current Summaries</h4>", unsafe_allow_html=True)
            
            # Show existing summaries in expandable sections if available
            col1, col2 = st.columns(2)
            
            with col1:
                if article.get('baseline_summary'):
                    with st.expander("Baseline Summary", expanded=False):
                        st.write(article.get('baseline_summary'))
                        if 'baseline_time' in article:
                            st.caption(f"Generated in {article.get('baseline_time', 0):.2f} seconds")
            
            with col2:
                if article.get('finetuned_summary'):
                    with st.expander("Fine-tuned Summary", expanded=False):
                        st.write(article.get('finetuned_summary'))
                        if 'finetuned_time' in article:
                            st.caption(f"Generated in {article.get('finetuned_time', 0):.2f} seconds")
        
        # Display keywords if available
        if article.get('keywords'):
            st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            st.markdown("<span style='color:#555; font-size:13px;'><b>Keywords:</b></span>", unsafe_allow_html=True)
            
            # Generate keyword tags with explicit color
            keywords_html = ""
            for kw in article.get('keywords', []):
                keywords_html += f'<span class="keyword-tag">{kw}</span>'
                
            st.markdown(f"<div style='margin-top:5px;'>{keywords_html}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Close the card div
        st.markdown('</div>', unsafe_allow_html=True)


def display_evaluation_metrics(eval_data: Dict[str, Any]) -> None:
    """
    Display evaluation metrics.
    
    Args:
        eval_data: Evaluation data dictionary
    """
    if not eval_data or 'summary' not in eval_data:
        st.warning("No evaluation data available")
        return
    
    summary = eval_data['summary']
    
    st.subheader("Evaluation Metrics")
    
    # ROUGE scores
    st.write("**ROUGE Scores:**")
    rouge_data = {
        'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
        'Baseline': [
            summary['baseline']['rouge1'],
            summary['baseline']['rouge2'],
            summary['baseline']['rougeL']
        ],
        'Fine-tuned': [
            summary['finetuned']['rouge1'],
            summary['finetuned']['rouge2'],
            summary['finetuned']['rougeL']
        ]
    }
    rouge_df = pd.DataFrame(rouge_data)
    rouge_df = rouge_df.set_index('Metric')
    st.bar_chart(rouge_df)
    
    # BLEU and BERTScore in columns
    col1, col2 = st.columns(2)
    
    # BLEU score
    with col1:
        st.write("**BLEU Score:**")
        bleu_data = {
            'Model': ['Baseline', 'Fine-tuned'],
            'Score': [
                summary['baseline']['bleu'],
                summary['finetuned']['bleu']
            ]
        }
        bleu_df = pd.DataFrame(bleu_data)
        bleu_df = bleu_df.set_index('Model')
        st.bar_chart(bleu_df)
    
    # BERTScore
    with col2:
        st.write("**BERTScore:**")
        bertscore_data = {
            'Model': ['Baseline', 'Fine-tuned'],
            'Score': [
                summary['baseline']['bertscore'],
                summary['finetuned']['bertscore']
            ]
        }
        bertscore_df = pd.DataFrame(bertscore_data)
        bertscore_df = bertscore_df.set_index('Model')
        st.bar_chart(bertscore_df)
    
    # Factuality
    # st.write("**Factuality Metrics:**")
    # factuality_data = {
    #     'Metric': ['Entailment', 'Contradiction', 'Hallucination Rate'],
    #     'Baseline': [
    #         summary['baseline']['entailment'],
    #         summary['baseline']['contradiction'],
    #         summary['baseline']['hallucination_rate']
    #     ],
    #     'Fine-tuned': [
    #         summary['finetuned']['entailment'],
    #         summary['finetuned']['contradiction'],
    #         summary['finetuned']['hallucination_rate']
    #     ]
    # }
    # factuality_df = pd.DataFrame(factuality_data)
    # factuality_df = factuality_df.set_index('Metric')
    # st.bar_chart(factuality_df)


def fetch_new_articles(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch and process new articles for a ticker.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        List of article dictionaries with summaries
    """
    st.info(f"Fetching new articles for {ticker}...")
    
    # Fetch articles
    articles = get_yahoo_finance_news(ticker, limit=5)
    
    # Track in app stats
    st.session_state.app_stats['articles_fetched'] += len(articles)
    st.session_state.app_stats['last_fetch_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add to recent activity
    add_activity(f"Fetched {len(articles)} articles for {ticker}")
    
    # Extract full text
    articles = extract_full_text(articles)
    
    # Clean text
    for i, article in enumerate(articles):
        if article.get('text'):
            articles[i]['cleaned_text'] = clean_text(article['text'])
            
        # Check if article already has a labeled summary (from the original source)
        # Yahoo Finance sometimes provides a summary in the article data
        if article.get('summary') and not article.get('labeled_summary'):
            articles[i]['labeled_summary'] = article['summary']
    
    # Generate summaries
    progress_bar = st.progress(0)
    
    # Load models
    baseline_model = setup_baseline_model(DEFAULT_BASELINE_MODEL)
    
    try:
        finetuned_model = setup_finetuned_model(DEFAULT_FINETUNED_PATH)
        has_finetuned = True
    except Exception as e:
        st.warning(f"Could not load fine-tuned model: {e}")
        has_finetuned = False
    
    # Generate summaries
    summaries_count = 0
    for i, article in enumerate(articles):
        text = article.get('cleaned_text', article.get('text', ''))
        
        if text:
            # Generate baseline summary
            try:
                result = baseline_model(
                    text,
                    max_length=150,
                    min_length=40,
                    do_sample=False
                )
                article['baseline_summary'] = result[0]['summary_text']
                summaries_count += 1
            except Exception as e:
                st.error(f"Error generating baseline summary: {e}")
            
            # Generate fine-tuned summary if available
            if has_finetuned:
                try:
                    result = finetuned_model(
                        text,
                        max_length=150,
                        min_length=40,
                        do_sample=False
                    )
                    print(result[0])
                    article['finetuned_summary'] = result[0]['summary_text']
                    summaries_count += 1
                except Exception as e:
                    st.error(f"Error generating fine-tuned summary: {e}")
                    
            # Categorize the article
            categorize_article(article)
        
        # Update progress
        progress_bar.progress((i + 1) / len(articles))
    
    progress_bar.empty()
    
    # Update summaries count in app stats
    st.session_state.app_stats['summaries_generated'] += summaries_count
    
    # Save articles
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = DATA_DIR / f"{ticker}_articles_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(articles, f, indent=2)
    
    st.success(f"Fetched and processed {len(articles)} articles for {ticker}")
    
    return articles


def categorize_article(article: Dict[str, Any]) -> None:
    """
    Categorize an article based on its content into predefined categories.
    This function adds a 'category' field to the article dictionary.
    
    Args:
        article: Article dictionary with text content
    """
    # Define category keywords
    categories = {
        "Earnings": [
            "earnings", "revenue", "profit", "EPS", "quarterly results", "financial results",
            "beat expectations", "missed expectations", "guidance", "outlook", "forecast",
            "dividend", "income", "loss", "earnings call", "earnings preview"
        ],
        "Product": [
            "launch", "new product", "unveil", "announce", "introduce", "release", "update",
            "version", "feature", "upgrade", "innovation", "patent", "prototype", "rollout"
        ],
        "M&A": [
            "acquisition", "merge", "takeover", "buyout", "purchase", "consolidation", "deal",
            "buy", "sell", "spin off", "spinoff", "divest", "joint venture", "partnership",
            "stake", "interest", "minority", "majority"
        ],
        "Leadership": [
            "CEO", "executive", "management", "board", "appoint", "hire", "promote", "resign",
            "step down", "successor", "leadership", "chief", "officer", "director", "chairman"
        ],
        "Legal": [
            "lawsuit", "litigation", "sue", "court", "settlement", "regulatory", "investigation",
            "probe", "fine", "penalty", "comply", "violation", "regulation", "appeal", "judge",
            "verdict", "patent", "intellectual property", "trademark", "copyright", "patent"
        ],
        "Market": [
            "stock", "share", "price", "market", "trading", "investor", "analyst", "downgrade",
            "upgrade", "rating", "recommendation", "target price", "valuation", "overweight",
            "underweight", "outperform", "neutral", "sell rating", "buy rating", "hold rating"
        ],
        "Economy": [
            "economy", "inflation", "recession", "growth", "GDP", "interest rate", "Fed",
            "Federal Reserve", "economic", "unemployment", "jobs", "labor", "consumer confidence",
            "slowdown", "recovery", "stimulus", "policy", "fiscal", "monetary", "demand", "supply"
        ]
    }
    
    # Default category
    article['category'] = "Other"
    
    # Combine title and text for searching
    search_text = " ".join([
        article.get('title', ''),
        article.get('description', ''),
        article.get('cleaned_text', article.get('text', ''))
    ]).lower()
    
    # Check each category
    category_scores = {category: 0 for category in categories}
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in search_text:
                # Count keyword occurrences and weight by importance
                # Title matches are more important than body text
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                
                # Different weights for different parts of the article
                if keyword.lower() in title:
                    category_scores[category] += 3  # Higher weight for title matches
                elif keyword.lower() in description:
                    category_scores[category] += 2  # Medium weight for description matches
                else:
                    category_scores[category] += 1  # Base weight for body matches
    
    # Assign the highest scoring category, if any
    if any(category_scores.values()):
        article['category'] = max(category_scores.items(), key=lambda x: x[1])[0]
    
    # Store category keywords found
    keywords_found = []
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in search_text:
                keywords_found.append(keyword)
    
    # Store up to 5 most relevant keywords
    article['keywords'] = keywords_found[:5]


def export_articles_to_csv(articles: List[Dict[str, Any]], filename: str = None) -> str:
    """
    Export articles to CSV format.
    
    Args:
        articles: List of article dictionaries
        filename: Optional filename to use
        
    Returns:
        CSV string of articles data
    """
    # Convert articles to a DataFrame
    rows = []
    for article in articles:
        # Extract the most important fields for CSV
        row = {
            'ticker': article.get('ticker', ''),
            'title': article.get('title', ''),
            'date': article.get('date', ''),
            'category': article.get('category', 'Other'),
            'source': article.get('source', ''),
            'url': article.get('url', ''),
            'baseline_summary': article.get('baseline_summary', ''),
            'finetuned_summary': article.get('finetuned_summary', ''),
            'labeled_summary': article.get('labeled_summary', article.get('reference_summary', article.get('ground_truth', ''))),
            'keywords': ', '.join(article.get('keywords', [])),
        }
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Convert to CSV
    csv_data = df.to_csv(index=False)
    
    return csv_data


def export_articles_to_excel(articles: List[Dict[str, Any]], filename: str = None) -> bytes:
    """
    Export articles to Excel format.
    
    Args:
        articles: List of article dictionaries
        filename: Optional filename to use
        
    Returns:
        Excel binary data
    """
    # Convert articles to a DataFrame
    rows = []
    for article in articles:
        # Extract the most important fields for Excel
        row = {
            'ticker': article.get('ticker', ''),
            'title': article.get('title', ''),
            'date': article.get('date', ''),
            'category': article.get('category', 'Other'),
            'source': article.get('source', ''),
            'url': article.get('url', ''),
            'baseline_summary': article.get('baseline_summary', ''),
            'finetuned_summary': article.get('finetuned_summary', ''),
            'labeled_summary': article.get('labeled_summary', article.get('reference_summary', article.get('ground_truth', ''))),
            'keywords': ', '.join(article.get('keywords', [])),
        }
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Convert to Excel
    excel_data = io.BytesIO()
    with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Articles', index=False)
    
    excel_data.seek(0)
    
    return excel_data.getvalue()


def export_articles_to_markdown(articles: List[Dict[str, Any]], filename: str = None) -> str:
    """
    Export articles to Markdown format.
    
    Args:
        articles: List of article dictionaries
        filename: Optional filename to use
        
    Returns:
        Markdown string of articles data
    """
    md_lines = ["# Financial News Summaries\n"]
    md_lines.append(f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Group articles by ticker
    ticker_articles = {}
    for article in articles:
        ticker = article.get('ticker', 'Unknown')
        if ticker not in ticker_articles:
            ticker_articles[ticker] = []
        ticker_articles[ticker].append(article)
    
    # Create markdown for each ticker
    for ticker, ticker_group in sorted(ticker_articles.items()):
        md_lines.append(f"\n## {ticker}\n")
        
        # Sort by date (newest first)
        ticker_group = sorted(ticker_group, key=lambda x: x.get('date', ''), reverse=True)
        
        for article in ticker_group:
            title = article.get('title', 'No Title')
            date = article.get('date', 'Unknown Date')
            category = article.get('category', 'Other')
            
            md_lines.append(f"### {title}\n")
            md_lines.append(f"**Date:** {date} | **Category:** {category}\n")
            
            if article.get('source'):
                md_lines.append(f"**Source:** {article.get('source')}\n")
                
            if article.get('url'):
                md_lines.append(f"**URL:** [{article.get('url')}]({article.get('url')})\n")
            
            if article.get('baseline_summary'):
                md_lines.append("\n**Baseline Summary:**\n")
                md_lines.append(f"{article.get('baseline_summary')}\n")
                
            if article.get('finetuned_summary'):
                md_lines.append("\n**Fine-tuned Summary:**\n")
                md_lines.append(f"{article.get('finetuned_summary')}\n")
                
            labeled_summary = article.get('labeled_summary') or article.get('reference_summary') or article.get('ground_truth')
            if labeled_summary:
                md_lines.append("\n**Labeled Summary:**\n")
                md_lines.append(f"{labeled_summary}\n")
                
            if article.get('keywords'):
                md_lines.append("\n**Keywords:** ")
                md_lines.append(', '.join(article.get('keywords', [])) + "\n")
                
            md_lines.append("\n---\n")
    
    return '\n'.join(md_lines)


def create_export_section(articles: List[Dict[str, Any]], title: str = "Export Data", key_prefix: str = ""):
    """
    Create an export section with buttons for various formats.
    
    Args:
        articles: List of article dictionaries to export
        title: Title for the export section
        key_prefix: Prefix for unique widget keys
    """
    with st.expander(title):
        if not articles:
            st.info("No articles available to export.")
            return
            
        st.write(f"Export {len(articles)} articles to your preferred format:")
        
        col1, col2, col3 = st.columns(3)
        
        # CSV export
        with col1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_data = export_articles_to_csv(articles)
            st.download_button(
                label="📄 Export as CSV",
                data=csv_data,
                file_name=f"news_summaries_{timestamp}.csv",
                mime="text/csv",
                key=f"{key_prefix}_csv_export"
            )
            st.caption("Spreadsheet format, good for data analysis")
        
        # Excel export
        with col2:
            excel_data = export_articles_to_excel(articles)
            st.download_button(
                label="📊 Export as Excel",
                data=excel_data,
                file_name=f"news_summaries_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key_prefix}_excel_export"
            )
            st.caption("Excel workbook with formatted data")
        
        # Markdown export
        with col3:
            md_data = export_articles_to_markdown(articles)
            st.download_button(
                label="📝 Export as Markdown",
                data=md_data,
                file_name=f"news_summaries_{timestamp}.md",
                mime="text/markdown",
                key=f"{key_prefix}_md_export"
            )
            st.caption("Text format for documentation/sharing")


def main():
    st.set_page_config(
        page_title="News Summarizer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 News Summarizer")
    
    # Load articles first
    articles = load_articles()
    
    # Initialize session state for tickers if not exists
    if 'available_tickers' not in st.session_state:
        st.session_state.available_tickers = sorted(list(set(article.get('ticker', '') for article in articles if article.get('ticker'))))
        
    # Initialize default portfolios if needed
    if 'tech_portfolio' not in st.session_state:
        st.session_state.tech_portfolio = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD", "TSLA"]
        
    if 'finance_portfolio' not in st.session_state:
        st.session_state.finance_portfolio = ["JPM", "BAC", "GS", "MS", "C", "WFC", "V", "MA", "AXP"]
        
    if 'energy_portfolio' not in st.session_state:
        st.session_state.energy_portfolio = ["XOM", "CVX", "BP", "SHEL", "COP", "SLB", "OXY", "EOG"]
    
    # Initialize recent activity tracking
    if 'recent_activity' not in st.session_state:
        st.session_state.recent_activity = []
        
    # Initialize app stats
    if 'app_stats' not in st.session_state:
        st.session_state.app_stats = {
            'articles_fetched': 0,
            'summaries_generated': 0,
            'last_fetch_date': None,
            'most_viewed_ticker': None
        }
        
    # Initialize usage counter for tickers
    if 'ticker_views' not in st.session_state:
        st.session_state.ticker_views = {}
    
    # Sidebar for all controls
    st.sidebar.title("Controls")
    
    # Portfolio Management in sidebar
    with st.sidebar.expander("Portfolio Settings", expanded=False):
        # Add ticker
        new_ticker = st.text_input("Add Ticker:", placeholder="e.g., AAPL")
        add_btn = st.button("Add to Portfolio")
        
        if add_btn and new_ticker:
            # Add ticker if it doesn't already exist
            new_ticker = new_ticker.upper()
            if new_ticker not in st.session_state.available_tickers:
                st.session_state.available_tickers.append(new_ticker)
                st.session_state.available_tickers.sort()
                st.sidebar.success(f"Added {new_ticker}")
                add_activity(f"Added {new_ticker} to portfolio")
            else:
                st.sidebar.info(f"{new_ticker} already exists")
    
    # Portfolio selection
    portfolio_options = {
        "My Portfolio": st.session_state.available_tickers,
        "Tech Stocks": st.session_state.tech_portfolio,
        "Financial Stocks": st.session_state.finance_portfolio,
        "Energy Stocks": st.session_state.energy_portfolio,
    }
    
    selected_portfolio = st.sidebar.selectbox("Portfolio:", list(portfolio_options.keys()))
    
    # Ticker selection in sidebar
    active_tickers = portfolio_options[selected_portfolio]
    
    # Handle case where active portfolio is empty
    if not active_tickers:
        st.sidebar.warning(f"Portfolio is empty!")
        ticker_options = [""]
    else:
        ticker_options = [""] + active_tickers
    
    selected_ticker = st.sidebar.selectbox(
        "Ticker:",
        ticker_options,
        index=0,
        format_func=lambda x: "All tickers" if x == "" else x
    )
    
    # Track ticker views when a specific ticker is selected
    if selected_ticker:
        if selected_ticker not in st.session_state.ticker_views:
            st.session_state.ticker_views[selected_ticker] = 0
        st.session_state.ticker_views[selected_ticker] += 1
        
        # Update most viewed ticker
        if not st.session_state.app_stats['most_viewed_ticker'] or \
           st.session_state.ticker_views[selected_ticker] > st.session_state.ticker_views.get(st.session_state.app_stats['most_viewed_ticker'], 0):
            st.session_state.app_stats['most_viewed_ticker'] = selected_ticker
    
    # Model selection
    st.sidebar.subheader("Summary Display")
    model_options = [
        "Baseline Model", 
        "Fine-tuned Model", 
        "Labeled/Ground Truth",
        "Generate New Summary"
    ]
    selected_model = st.sidebar.radio(
        "Display:",
        model_options,
        index=0
    )
    
    # Fetch control - simplified
    st.sidebar.subheader("Data Refresh")
    if st.sidebar.button("📰 Fetch News"):
        if selected_ticker:
            with st.spinner(f"Fetching articles for {selected_ticker}..."):
                new_articles = fetch_new_articles(selected_ticker)
                articles.extend(new_articles)
                st.success(f"Fetched {len(new_articles)} articles for {selected_ticker}")
                add_activity(f"Fetched {len(new_articles)} articles for {selected_ticker}")
        else:
            active_tickers = portfolio_options[selected_portfolio]
            if active_tickers:
                with st.spinner(f"Fetching articles for portfolio..."):
                    total_new_articles = 0
                    for ticker in active_tickers:
                        new_articles = fetch_new_articles(ticker)
                        articles.extend(new_articles)
                        total_new_articles += len(new_articles)
                    st.success(f"Fetched {total_new_articles} articles")
    
    # Add tabbed interface for different sections - simplified to 3 tabs
    tabs = st.tabs(["News", "Fine-tuning", "Evaluation"])
    
    # News tab (first tab)
    with tabs[0]:
        st.header("News Articles")
        
        # Filter articles by selected ticker
        filtered_articles = articles
        if selected_ticker:
            filtered_articles = filter_articles_by_ticker(articles, selected_ticker)
        
        # Check if we have articles to display
        if not filtered_articles:
            st.info(f"No articles found for {selected_ticker if selected_ticker else 'your portfolio'}. Click 'Fetch News' to get the latest news.")
        
        # Display articles
        num_articles = len(filtered_articles)
        if num_articles > 0:
            st.write(f"Found {num_articles} articles")
            
            # If "Generate New Summary" was selected, show model selection for inference
            if selected_model == "Generate New Summary":
                # Inference model selection
                st.subheader("Generate New Summaries")
                inference_col1, inference_col2 = st.columns(2)
                
                with inference_col1:
                    baseline_model_name = st.text_input("Baseline Model:", 
                                                      value=DEFAULT_BASELINE_MODEL,
                                                      help="HuggingFace model ID (e.g., facebook/bart-large-cnn)")
                
                with inference_col2:
                    finetuned_model_path = st.text_input("Fine-tuned Model Path:", 
                                                       value=DEFAULT_FINETUNED_PATH,
                                                       help="Path to fine-tuned model")
                
                # Generate button
                if st.button("Generate Summaries for Selected Articles"):
                    with st.spinner("Setting up models..."):
                        # Set up models for inference
                        try:
                            baseline_model = setup_baseline_model(baseline_model_name)
                            finetuned_model = setup_finetuned_model(finetuned_model_path)
                            
                            # Process articles
                            progress_bar = st.progress(0)
                            
                            for i, article in enumerate(filtered_articles):
                                # Get the text to summarize
                                text = article.get("cleaned_text", article.get("text", ""))
                                
                                if text:
                                    # Generate baseline summary
                                    start_time = time.time()
                                    baseline_result = baseline_model(
                                        text,
                                        max_length=150,
                                        min_length=40,
                                        do_sample=False
                                    )
                                    baseline_time = time.time() - start_time
                                    
                                    article["baseline_summary"] = baseline_result[0]["summary_text"]
                                    article["baseline_time"] = baseline_time
                                    
                                    # Generate fine-tuned summary
                                    start_time = time.time()
                                    finetuned_result = finetuned_model(
                                        text,
                                        max_length=150,
                                        min_length=40,
                                        do_sample=False
                                    )
                                    finetuned_time = time.time() - start_time
                                    
                                    article["finetuned_summary"] = finetuned_result[0]["summary_text"]
                                    article["finetuned_time"] = finetuned_time
                                
                                # Update progress
                                progress_bar.progress((i+1) / len(filtered_articles))
                            
                            st.success(f"Generated summaries for {len(filtered_articles)} articles")
                            
                            # Update stats
                            st.session_state.app_stats['summaries_generated'] += len(filtered_articles)
                            add_activity(f"Generated summaries for {len(filtered_articles)} articles")
                            
                        except Exception as e:
                            st.error(f"Error generating summaries: {str(e)}")
            
            # Display articles
            for i, article in enumerate(filtered_articles):
                with st.container():
                    st.write("---")
                    display_article(article, i, selected_model)
            
            # Add export options at the bottom
            create_export_section(filtered_articles)
    
    # Fine-tuning tab
    with tabs[1]:
        st.header("Model Fine-tuning")
        
        # Datasets section
        st.subheader("Dataset Configuration")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Hugging Face Dataset", "Local JSON File", "Current Articles"],
            horizontal=True
        )
        
        if data_source == "Hugging Face Dataset":
            col1, col2 = st.columns(2)
            with col1:
                dataset_name = st.text_input("Dataset Name:", placeholder="e.g., cnn_dailymail, xsum")
                text_column = st.text_input("Text Column:", value="text")
                split = st.selectbox("Dataset Split:", ["train", "validation", "test"])
            with col2:
                summary_column = st.text_input("Summary Column:", value="summary")
                max_samples = st.number_input("Max Samples (0 for all):", min_value=0, value=1000)
                filter_query = st.text_input("Filter Query:", placeholder="Optional dataset filter (HF syntax)")
                
        elif data_source == "Local JSON File":
            uploaded_file = st.file_uploader("Upload JSON file:", type=["json"])
            
            col1, col2 = st.columns(2)
            with col1:
                text_field = st.text_input("Text Field Name:", value="text")
                train_split = st.slider("Training Split:", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
            with col2:
                summary_field = st.text_input("Summary Field Name:", value="summary")
                val_split = st.slider("Validation Split:", min_value=0.0, max_value=0.3, value=0.1, step=0.05)
                
        elif data_source == "Current Articles":
            st.info("Uses currently loaded articles for fine-tuning.")
            
            min_length = st.slider("Minimum Article Length:", min_value=100, max_value=2000, value=500, step=100)
            require_summary = st.checkbox("Require Reference Summary", value=True, 
                                         help="Only use articles with reference summaries")
        
        # Model Configuration section
        st.subheader("Model Configuration")
        
        tabs_model = st.tabs(["Base Model", "LoRA Parameters", "Advanced Configuration"])
        
        with tabs_model[0]:
            col1, col2 = st.columns(2)
            with col1:
                model_options = [
                    "facebook/bart-large-cnn",
                    "google/pegasus-xsum",
                    "t5-small",
                    "t5-base",
                    "t5-large",
                    "sshleifer/distilbart-cnn-12-6",
                    "philschmid/bart-large-cnn-samsum",
                    "Other"
                ]
                base_model = st.selectbox("Base Model:", model_options)
                
                if base_model == "Other":
                    base_model = st.text_input("Custom Model Name:", placeholder="Enter HuggingFace model name")
                
                use_peft = st.checkbox("Use PEFT (Parameter-Efficient Fine-Tuning)", value=True)
                peft_type = st.selectbox("PEFT Method:", ["LoRA", "Prefix Tuning", "Prompt Tuning", "P-Tuning"], 
                                        disabled=not use_peft)
            
            with col2:
                output_dir = st.text_input("Output Directory:", value="models/finetuned/custom")
                quantization = st.selectbox("Quantization:", ["None", "4-bit", "8-bit"])
                device = st.selectbox("Training Device:", ["cuda", "cpu", "mps"], 
                                      index=0 if torch.cuda.is_available() else 2 if torch.backends.mps.is_available() else 1)
        
        with tabs_model[1]:
            # LoRA specific parameters
            col1, col2 = st.columns(2)
            with col1:
                lora_r = st.slider("LoRA Rank (r):", min_value=1, max_value=256, value=16, step=1)
                lora_alpha = st.slider("LoRA Alpha:", min_value=1, max_value=256, value=32, step=1)
                
                target_modules = st.text_input("Target Modules:", 
                                             placeholder="comma-separated, e.g., q_proj,v_proj,k_proj")
            
            with col2:
                lora_dropout = st.slider("LoRA Dropout:", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
                lora_bias = st.selectbox("LoRA Bias:", ["none", "all", "lora_only"])
                fan_in_fan_out = st.checkbox("Fan-in/Fan-out", value=False)
        
        with tabs_model[2]:
            # Advanced model settings
            col1, col2 = st.columns(2)
            with col1:
                max_length = st.slider("Max Source Length:", min_value=128, max_value=2048, value=512, step=64)
                max_target_length = st.slider("Max Target Length:", min_value=32, max_value=512, value=128, step=32)
                padding = st.selectbox("Padding Strategy:", ["max_length", "longest", "do_not_pad"])
            
            with col2:
                truncation = st.checkbox("Truncation", value=True)
                num_beams = st.slider("Num Beams:", min_value=1, max_value=10, value=4, step=1)
                use_flash_attention = st.checkbox("Use Flash Attention", value=False)
        
        # Training Parameters section
        st.subheader("Training Parameters")
        
        tabs_train = st.tabs(["Basic", "Optimization", "Scheduling"])
        
        with tabs_train[0]:
            col1, col2, col3 = st.columns(3)
            with col1:
                num_epochs = st.slider("Number of Epochs:", min_value=1, max_value=20, value=3, step=1)
                batch_size = st.slider("Training Batch Size:", min_value=1, max_value=64, value=8, step=1)
            
            with col2:
                gradient_accumulation = st.slider("Gradient Accumulation Steps:", min_value=1, max_value=16, value=1, step=1)
                eval_steps = st.number_input("Evaluation Steps:", min_value=0, value=500, help="0 means evaluate at end of epoch")
            
            with col3:
                fp16 = st.checkbox("Mixed Precision (FP16)", value=True)
                eval_batch_size = st.slider("Evaluation Batch Size:", min_value=1, max_value=64, value=8, step=1)
        
        with tabs_train[1]:
            col1, col2 = st.columns(2)
            with col1:
                learning_rate = st.number_input("Learning Rate:", min_value=1e-6, max_value=1e-2, value=2e-5, format="%.1e")
                weight_decay = st.number_input("Weight Decay:", min_value=0.0, max_value=0.2, value=0.01, step=0.01)
                adam_beta1 = st.slider("Adam Beta1:", min_value=0.5, max_value=0.999, value=0.9, step=0.001, format="%.3f")
            
            with col2:
                adam_beta2 = st.slider("Adam Beta2:", min_value=0.9, max_value=0.9999, value=0.999, step=0.0001, format="%.4f")
                adam_epsilon = st.number_input("Adam Epsilon:", min_value=1e-10, max_value=1e-6, value=1e-8, format="%.1e")
                max_grad_norm = st.slider("Gradient Clipping:", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        with tabs_train[2]:
            col1, col2 = st.columns(2)
            with col1:
                lr_scheduler_type = st.selectbox("LR Scheduler:", 
                                              ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
                warmup_steps = st.number_input("Warmup Steps:", min_value=0, value=500)
            
            with col2:
                early_stopping = st.checkbox("Early Stopping", value=True)
                patience = st.number_input("Patience:", min_value=1, value=3, disabled=not early_stopping)
                early_stopping_threshold = st.number_input("Improvement Threshold:", min_value=0.0, value=0.01, step=0.001, disabled=not early_stopping)
        
        # Logging and Monitoring
        st.subheader("Logging and Monitoring")
        
        col1, col2 = st.columns(2)
        with col1:
            use_wandb = st.checkbox("Use Weights & Biases", value=True)
            if use_wandb:
                wandb_project = st.text_input("W&B Project Name:", value="news-summarizer")
                wandb_name = st.text_input("W&B Run Name:", placeholder="Optional, will be auto-generated if empty")
        
        with col2:
            save_strategy = st.selectbox("Save Strategy:", ["steps", "epoch", "no"])
            save_steps = st.number_input("Save Steps:", min_value=0, value=500, disabled=save_strategy != "steps")
            save_total_limit = st.number_input("Save Total Limit:", min_value=1, value=3, 
                                             help="Maximum number of checkpoints to keep")
        
        # Submit button
        if st.button("Start Fine-tuning", type="primary"):
            # This would trigger the fine-tuning process
            st.info("Fine-tuning would start here with the selected parameters. This feature is not yet fully implemented in the UI.")
            
            # Display the command that would be executed
            st.subheader("Equivalent Command")
            
            # Build command based on selections
            cmd = ["python finetune.py"]
            
            # Add base model
            cmd.append(f"--model \"{base_model}\"")
            
            # Add data source
            if data_source == "Hugging Face Dataset":
                cmd.append(f"--dataset-name \"{dataset_name}\"")
                cmd.append(f"--text-column \"{text_column}\"")
                cmd.append(f"--summary-column \"{summary_column}\"")
                cmd.append(f"--split \"{split}\"")
                if max_samples > 0:
                    cmd.append(f"--max-samples {max_samples}")
                if filter_query:
                    cmd.append(f"--filter-query \"{filter_query}\"")
            
            # Add output directory
            cmd.append(f"--output-dir \"{output_dir}\"")
            
            # Add training params
            cmd.append(f"--num-epochs {num_epochs}")
            cmd.append(f"--batch-size {batch_size}")
            cmd.append(f"--learning-rate {learning_rate}")
            
            # Add LoRA params if using PEFT
            if use_peft and peft_type == "LoRA":
                cmd.append("--use-peft")
                cmd.append(f"--lora-r {lora_r}")
                cmd.append(f"--lora-alpha {lora_alpha}")
                cmd.append(f"--lora-dropout {lora_dropout}")
            
            # Add W&B if selected
            if use_wandb:
                cmd.append("--use-wandb")
                cmd.append(f"--wandb-project \"{wandb_project}\"")
                if wandb_name:
                    cmd.append(f"--wandb-name \"{wandb_name}\"")
            
            # Display the command
            st.code(" ".join(cmd))
    
    # Evaluation tab
    with tabs[2]:
        st.header("Model Evaluation")
        
        # Load evaluation data if available
        eval_data = load_evaluation_data()
        
        if not eval_data:
            st.info("No evaluation data available. Run evaluation metrics first.")
        else:
            display_evaluation_metrics(eval_data)


def add_activity(description: str) -> None:
    """
    Add an entry to the recent activity log.
    
    Args:
        description: Description of the activity
    """
    # Get current time
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create activity entry
    activity = {
        "time": now,
        "description": description
    }
    
    # Add to activity log (max 10 entries)
    st.session_state.recent_activity.insert(0, activity)
    st.session_state.recent_activity = st.session_state.recent_activity[:10]


if __name__ == "__main__":
    main() 