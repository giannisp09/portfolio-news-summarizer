#!/usr/bin/env python3
"""
Script to clean and preprocess news articles.
- Strips HTML/boilerplate
- Normalizes whitespace
- Segments into sentences
- Tokenizes text
- Deduplicates articles
"""

import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
import re

import spacy
from tqdm import tqdm

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def load_articles(input_file: str) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.
    
    Args:
        input_file: Path to the JSON file containing articles
        
    Returns:
        List of article dictionaries
    """
    with open(input_file, 'r') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {input_file}")
    return articles


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, normalizing whitespace, etc.
    
    Args:
        text: Raw article text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^\w\s.,?!;:\-\'"]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def segment_and_tokenize(text: str, nlp) -> Dict[str, Any]:
    """
    Segment text into sentences and tokenize.
    
    Args:
        text: Cleaned article text
        nlp: spaCy language model
        
    Returns:
        Dictionary with sentences and tokens
    """
    if not text:
        return {"sentences": [], "tokens": []}
    
    doc = nlp(text)
    
    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Extract tokens (excluding punctuation and whitespace)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    
    return {
        "sentences": sentences,
        "tokens": tokens
    }


def generate_content_hash(title: str, text: str) -> str:
    """
    Generate a hash of article content to identify duplicates.
    
    Args:
        title: Article title
        text: Article text
        
    Returns:
        Hash string
    """
    content = (title + " " + text).lower()
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate articles based on URL or content hash.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Deduplicated list of article dictionaries
    """
    unique_urls = set()
    unique_hashes = set()
    deduplicated = []
    
    for article in articles:
        url = article.get("url", "")
        title = article.get("title", "")
        text = article.get("text", "")
        
        # Generate content hash
        content_hash = generate_content_hash(title, text)
        
        # Check if this is a duplicate
        if url in unique_urls or content_hash in unique_hashes:
            continue
            
        # Add to deduplicated list and update tracking sets
        deduplicated.append(article)
        
        if url:
            unique_urls.add(url)
            
        unique_hashes.add(content_hash)
        
    print(f"Removed {len(articles) - len(deduplicated)} duplicate articles")
    return deduplicated


def process_articles(articles: List[Dict[str, Any]], nlp) -> List[Dict[str, Any]]:
    """
    Clean and process all articles.
    
    Args:
        articles: List of article dictionaries
        nlp: spaCy language model
        
    Returns:
        List of processed article dictionaries
    """
    processed_articles = []
    
    for article in tqdm(articles, desc="Processing articles"):
        # Skip articles without text
        if not article.get("text"):
            continue
            
        # Clean the text
        cleaned_text = clean_text(article["text"])
        
        # Skip articles with insufficient text
        if len(cleaned_text.split()) < 20:  # Arbitrary minimum length
            continue
            
        # Segment and tokenize
        processed_text = segment_and_tokenize(cleaned_text, nlp)
        
        # Create processed article
        processed_article = article.copy()
        processed_article["cleaned_text"] = cleaned_text
        processed_article["sentences"] = processed_text["sentences"]
        processed_article["tokens"] = processed_text["tokens"]
        
        processed_articles.append(processed_article)
        
    return processed_articles


def save_processed_articles(articles: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save processed articles to a JSON file.
    
    Args:
        articles: List of processed article dictionaries
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(articles, f, indent=2)
        
    print(f"Saved {len(articles)} processed articles to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess news articles")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file containing scraped articles")
    parser.add_argument("--output", type=str, help="Output JSON file for processed articles")
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(DATA_DIR / f"{input_path.stem}_cleaned.json")
    
    # Load spaCy model
    print("Loading language model...")
    nlp = spacy.load("en_core_web_sm")
    
    # Load articles
    articles = load_articles(args.input)
    
    # Process articles
    processed_articles = process_articles(articles, nlp)
    
    # Deduplicate articles
    deduplicated_articles = deduplicate_articles(processed_articles)
    
    # Save processed articles
    save_processed_articles(deduplicated_articles, args.output)


if __name__ == "__main__":
    main() 