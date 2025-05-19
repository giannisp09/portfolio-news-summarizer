#!/usr/bin/env python3
"""
Script to gather a large dataset of news articles for multiple tickers.
This script automates the process of scraping, cleaning, and preparing data
for both training and evaluation.
"""

import os
import json
import argparse
import time
from pathlib import Path
import random
from typing import List, Dict, Any
import concurrent.futures
import sys

# Import project modules
from scraper import get_yahoo_finance_news, extract_full_text, save_articles
from cleaner import clean_text, process_articles, deduplicate_articles
import spacy

# Ensure necessary directories exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Default tickers - a mix of major tech, finance, automotive, retail, and other sectors
DEFAULT_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "INTC", "AMD", "ORCL", "IBM", 
    "CRM", "ADBE", "CSCO", "TSM", "QCOM",
    
    # Finance
    "JPM", "BAC", "GS", "WFC", "C", "MS", "AXP", "V", "MA", "PYPL", "BLK", "SCHW",
    
    # Automotive
    "TSLA", "F", "GM", "TM", "HMC", "STLA", "RACE", "RIVN", "LCID",
    
    # Retail
    "WMT", "TGT", "COST", "HD", "LOW", "AMZN", "BABA", "JD", "MELI", "EBAY", "ETSY",
    
    # Energy
    "XOM", "CVX", "COP", "BP", "SHEL", "TTE", "ENB", "ET", "NEE", "DUK",
    
    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "UNH", "CVS", "WBA", "GILD", "MRNA", "REGN",
    
    # Communication
    "VZ", "T", "TMUS", "CMCSA", "CHTR", "NFLX", "DIS", "ROKU", "TTD",
    
    # Index ETFs (for market news)
    "SPY", "QQQ", "DIA", "IWM"
]

# Additional tickers to expand data collection
OTHER_TICKERS = [
    # More Tech
    "ADSK", "AMAT", "AVGO", "CDNS", "CTSH", "DDOG", "DBX", "DOCU", "FIVN", "FTNT", 
    "INTU", "LRCX", "MU", "NOW", "NET", "NICE", "OKTA", "PANW", "PLTR", "PTC", 
    "SNPS", "TEAM", "TXN", "TWLO", "UBER", "VMW", "WDAY", "ZM", "ZS",
    
    # More Finance
    "AFL", "AIG", "AXP", "BEN", "BK", "BLK", "BX", "CB", "CINF", "CMA", "COF", 
    "DFS", "FITB", "HIG", "HBAN", "ICE", "IVZ", "KEY", "L", "MTB", "MET", 
    "NTRS", "PBCT", "PGR", "PNC", "PRU", "RF", "SPGI", "STT", "TROW", "TRV", 
    "USB", "WFC", "ZION",
    
    # More Consumer & Retail
    "AAP", "BBWI", "BBY", "BURL", "CPRI", "DG", "DKS", "DLTR", "DPZ", "EL", 
    "ETSY", "FL", "GPC", "GPS", "JWN", "KMX", "KR", "LB", "M", "MCD", 
    "MNST", "ORLY", "ROST", "SBUX", "TGT", "TJX", "TSCO", "UAA", "ULTA", "VFC", 
    "WMT", "WSM", "YUM",
    
    # International
    "SONY", "HMC", "TM", "NSANY", "SNE", "NTDOY", "TCEHY", "BABA", "JD", "NTES", 
    "PDD", "MELI", "SE", "SAP", "UL", "BHP", "RIO", "BP", "TOT", "SHOP", "ABB",
    
    # Crypto-related
    "COIN", "SQ", "MARA", "RIOT", "SI", "MSTR",
    
    # Semiconductor focused
    "MCHP", "MPWR", "NXPI", "ON", "SWKS", "TER", "XLNX",
    
    # AI focused
    "BBAI", "UPST", "PATH", "AI", "CRWD", "MDB", "SNOW"
]


def get_articles_for_ticker(ticker: str, 
                           limit: int = 50, 
                           delay: int = 5, 
                           extract_text: bool = True,
                           nlp = None,
                           start_date: str = None,
                           end_date: str = None,
                           max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Scrape and process articles for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        limit: Number of articles to retrieve
        delay: Delay between requests to avoid rate limits
        extract_text: Whether to extract full text
        nlp: Optional spaCy model for processing
        max_retries: Maximum number of retries on failure
        
    Returns:
        List of processed article dictionaries
    """
    print(f"Processing ticker {ticker}...")
    
    for retry in range(max_retries + 1):
        try:
            # Get articles from Yahoo Finance
            articles = get_yahoo_finance_news(ticker, limit, start_date, end_date)
            
            # Skip if no articles found
            if not articles:
                print(f"No articles found for {ticker}")
                return []
                
            print(f"Found {len(articles)} articles for {ticker}")
            
            # Extract full text if requested
            if extract_text:
                try:
                    start_time = time.time()
                    articles = extract_full_text(articles)
                    print(f"Extracted full text for {ticker} in {time.time() - start_time:.2f} seconds")
                except Exception as text_error:
                    print(f"Error extracting full text for {ticker}: {str(text_error)}")
                    # Continue with the articles we have, even if text extraction failed
                
            # Clean and process articles if nlp is provided
            if nlp is not None:
                # Clean each article
                for i, article in enumerate(articles):
                    if article.get("text"):
                        try:
                            articles[i]["cleaned_text"] = clean_text(article["text"])
                        except Exception as clean_error:
                            print(f"Error cleaning text for article {i} of {ticker}: {str(clean_error)}")
                            # Keep the original text
                            articles[i]["cleaned_text"] = article["text"]
                
                # Further process with nlp
                try:
                    articles = process_articles(articles, nlp)
                except Exception as nlp_error:
                    print(f"Error in NLP processing for {ticker}: {str(nlp_error)}")
                    # Continue without NLP processing
                
            # Deduplicate articles
            original_count = len(articles)
            articles = deduplicate_articles(articles)
            if len(articles) < original_count:
                print(f"Removed {original_count - len(articles)} duplicate articles for {ticker}")
            
            # Add delay to avoid hitting rate limits
            time.sleep(delay)
            
            return articles
            
        except Exception as e:
            if retry < max_retries:
                retry_delay = delay * (retry + 1)  # Exponential backoff
                print(f"Error processing ticker {ticker} (attempt {retry+1}/{max_retries+1}): {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to process ticker {ticker} after {max_retries+1} attempts: {str(e)}")
                return []


def gather_articles(tickers: List[str], 
                   articles_per_ticker: int = 20,
                   max_workers: int = 3, 
                   extract_text: bool = True,
                   start_date: str = None,
                   end_date: str = None,
                   clean_text: bool = True) -> List[Dict[str, Any]]:
    """
    Gather articles for multiple tickers, potentially in parallel.
    
    Args:
        tickers: List of ticker symbols
        articles_per_ticker: Number of articles to retrieve per ticker
        max_workers: Maximum number of parallel workers (use with caution)
        extract_text: Whether to extract full text
        clean_text: Whether to clean and process text
        
    Returns:
        List of processed article dictionaries
    """
    all_articles = []
    
    # Load spaCy model if text cleaning is requested
    nlp = None
    if clean_text:
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
    
    # Single-threaded version (safer but slower)
    if max_workers <= 1:
        for ticker in tickers:
            articles = get_articles_for_ticker(
                ticker, 
                limit=articles_per_ticker,
                extract_text=extract_text,
                nlp=nlp,
                start_date=start_date,
                end_date=end_date
            )
            all_articles.extend(articles)
    
    # Multi-threaded version (faster but use with caution)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    get_articles_for_ticker, 
                    ticker, 
                    articles_per_ticker,
                    10,  # Increased delay for parallel execution 
                    extract_text,
                    nlp
                ): ticker for ticker in tickers
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                    print(f"Completed processing {ticker}, total articles: {len(all_articles)}")
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
    
    return all_articles


def split_for_training(articles: List[Dict[str, Any]], 
                      train_ratio: float = 0.8,
                      shuffle: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split the articles into training and evaluation sets.
    
    Args:
        articles: List of article dictionaries
        train_ratio: Ratio of data to use for training
        shuffle: Whether to shuffle the data before splitting
        
    Returns:
        Dictionary with 'train' and 'eval' keys containing the split data
    """
    # Work with a copy of the list to avoid modifying the original
    working_articles = articles.copy()
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(working_articles)
    
    # Calculate split point
    split_idx = int(len(working_articles) * train_ratio)
    
    # Split the data
    return {
        'train': working_articles[:split_idx],
        'eval': working_articles[split_idx:]
    }


def main():
    parser = argparse.ArgumentParser(description="Gather a large dataset of news articles for multiple tickers")
    parser.add_argument("--tickers", type=str, nargs="+", help="List of ticker symbols to process")
    parser.add_argument("--tickers-file", type=str, help="File containing ticker symbols, one per line")
    parser.add_argument("--articles-per-ticker", type=int, default=50, help="Number of articles to retrieve per ticker")
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "multi_ticker_dataset.json"), help="Output file path")
    parser.add_argument("--no-extract-text", action="store_true", help="Skip full text extraction")
    parser.add_argument("--no-clean-text", action="store_true", help="Skip text cleaning and processing")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of parallel workers")
    parser.add_argument("--no-split", action="store_true", help="Don't split into train/eval sets")
    parser.add_argument("--use-all-tickers", action="store_true", help="Use both DEFAULT_TICKERS and OTHER_TICKERS")
    parser.add_argument("--delay", type=int, default=5, help="Delay between requests in seconds")
    parser.add_argument("--batch-size", type=int, default=10, help="Process tickers in batches of this size")
    parser.add_argument("--start-date", type=str, default=None, help="Start date for article retrieval")
    parser.add_argument("--end-date", type=str, default=None, help="End date for article retrieval")
    
    args = parser.parse_args()
    
    # Determine which tickers to process
    tickers = []
    
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]
    elif args.tickers_file:
        with open(args.tickers_file, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    elif args.use_all_tickers:
        # Use both default and other tickers for maximum data collection
        tickers = DEFAULT_TICKERS + OTHER_TICKERS
        # Remove duplicates while preserving order
        tickers = list(dict.fromkeys(tickers))
    else:
        tickers = DEFAULT_TICKERS
        
    print(f"Processing {len(tickers)} tickers: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
    
    # For large collections, process in batches to avoid memory issues
    all_articles = []
    
    if args.batch_size and len(tickers) > args.batch_size:
        print(f"Processing tickers in batches of {args.batch_size}")
        
        # Split tickers into batches
        ticker_batches = [tickers[i:i + args.batch_size] for i in range(0, len(tickers), args.batch_size)]
        
        for i, batch in enumerate(ticker_batches):
            print(f"Processing batch {i+1}/{len(ticker_batches)} with {len(batch)} tickers")
            
            # Process this batch
            batch_articles = gather_articles(
                batch,
                articles_per_ticker=args.articles_per_ticker,
                max_workers=args.max_workers,
                extract_text=not args.no_extract_text,
                clean_text=not args.no_clean_text
            )
            
            # Add to overall collection
            all_articles.extend(batch_articles)
            
            # Save intermediate results
            intermediate_output = Path(str(args.output).replace(".json", f"_batch_{i+1}.json"))
            with open(intermediate_output, 'w') as f:
                json.dump(batch_articles, f, indent=2)
            
            print(f"Saved {len(batch_articles)} articles from batch {i+1} to {intermediate_output}")
            print(f"Total articles so far: {len(all_articles)}")
            
            # Sleep between batches to avoid rate limiting
            if i < len(ticker_batches) - 1:
                sleep_time = args.delay * 2  # Longer delay between batches
                print(f"Sleeping for {sleep_time} seconds before next batch...")
                time.sleep(sleep_time)
    else:
        # Process all tickers in one go
        start_time = time.time()
        all_articles = gather_articles(
            tickers,
            articles_per_ticker=args.articles_per_ticker,
            max_workers=args.max_workers,
            extract_text=not args.no_extract_text,
            clean_text=not args.no_clean_text
        )
        print(f"Gathered {len(all_articles)} total articles in {time.time() - start_time:.2f} seconds")
    
    # Save full dataset
    output_file = Path(args.output)
    output_dir = output_file.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.no_split:
        # Save as a single file
        with open(output_file, 'w') as f:
            json.dump(all_articles, f, indent=2)
        print(f"Saved {len(all_articles)} articles to {output_file}")
    else:
        # Split and save as separate files
        dataset = split_for_training(all_articles, train_ratio=args.train_ratio)
        
        # Create output filenames
        base_name = output_file.stem
        train_file = output_dir / f"{base_name}_train.json"
        eval_file = output_dir / f"{base_name}_eval.json"
        
        # Save split datasets
        with open(train_file, 'w') as f:
            json.dump(dataset['train'], f, indent=2)
            
        with open(eval_file, 'w') as f:
            json.dump(dataset['eval'], f, indent=2)
            
        print(f"Saved {len(dataset['train'])} articles to {train_file}")
        print(f"Saved {len(dataset['eval'])} articles to {eval_file}")
    
    print("Done!")


if __name__ == "__main__":
    main() 