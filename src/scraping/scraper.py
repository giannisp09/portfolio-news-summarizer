#!/usr/bin/env python3
"""
Script to scrape news articles for a given stock ticker symbol.
Supports multiple sources: Yahoo Finance, NewsAPI, and direct web scraping with newspaper3k.
"""

import os
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

import requests
import yfinance as yf
from newspaper import Article
from newspaper import fulltext as extract_fulltext
from dotenv import load_dotenv
# from newsapi import NewsApiClient

# Load environment variables (API keys)
load_dotenv()

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def get_yahoo_finance_news(ticker: str, limit: int = 10, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    """
    Get news from Yahoo Finance for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "TSLA")
        limit: Maximum number of articles to retrieve
        
    Returns:
        List of news article dictionaries
    """
    print(f"Fetching Yahoo Finance news for {ticker}...")
    
    # Get ticker information
    stock = yf.Ticker(ticker)
    news = stock.news
    
    # Debug: Print structure of first news item
    if news and len(news) > 0:
        print(f"News item structure keys: {list(news[0].keys())}")
    
    # Limit results
    news = news[:limit]
    
    articles = []
    for item in news:
        try:
            # Extract article content safely
            content = item.get('content', {}) if item else {}
            
            # Get URL safely with fallbacks
            url = ""
            if content and content.get("clickThroughUrl"):
                url = content.get("clickThroughUrl", {}).get("url", "")
            elif content and content.get("canonicalUrl"):
                url = content.get("canonicalUrl", {}).get("url", "")
            elif item.get("link"):
                url = item.get("link", "")
                
            # Get date safely with fallbacks
            date = ""
            if content and content.get("pubDate"):
                date = content.get("pubDate", "")
            elif item.get("providerPublishTime"):
                try:
                    date = datetime.datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat()
                except:
                    date = ""
                    
            # Get provider safely
            provider = "Yahoo Finance"
            if content and content.get("provider") and content.get("provider", {}).get("displayName"):
                provider = content.get("provider", {}).get("displayName", "Yahoo Finance")
                
            # Create article data
            article_data = {
                "ticker": ticker,
                "title": content.get("title", "") if content else item.get("title", ""),
                "date": date,
                "url": url,
                "source": provider,
                "summary": content.get("summary", "") if content else "",
                "text": None  # Will be filled in by extract_full_text
            }
            
            articles.append(article_data)
            
        except Exception as e:
            print(f"Error processing news item: {str(e)}")
            print(f"Item data: {item}")
        
    return articles


# def get_newsapi_articles(ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
#     """
#     Get news from NewsAPI for a specific ticker.
#     
#     Args:
#         ticker: Stock ticker symbol (e.g., "TSLA")
#         limit: Maximum number of articles to retrieve
#         
#     Returns:
#         List of news article dictionaries
#     """
#     api_key = os.getenv("NEWSAPI_KEY")
#     if not api_key:
#         print("Warning: NEWSAPI_KEY not found in environment variables. Skipping NewsAPI.")
#         return []
#     
#     print(f"Fetching NewsAPI articles for {ticker}...")
#     
#     try:
#         from newsapi import NewsApiClient
#         newsapi = NewsApiClient(api_key=api_key)
#         
#         # Search for articles containing the ticker
#         response = newsapi.get_everything(
#             q=f"{ticker} stock",
#             language='en',
#             sort_by='publishedAt',
#             page_size=limit
#         )
#         
#         articles = []
#         for item in response.get('articles', []):
#             article_data = {
#                 "ticker": ticker,
#                 "title": item.get("title", ""),
#                 "date": item.get("publishedAt", ""),
#                 "url": item.get("url", ""),
#                 "source": f"NewsAPI - {item.get('source', {}).get('name', 'Unknown')}",
#                 "summary": item.get("description", ""),
#                 "text": item.get("content", "")  # This is usually truncated and needs to be expanded
#             }
#             articles.append(article_data)
#             
#         return articles
#     except ImportError:
#         print("NewsAPI Python client not installed. Install with: pip install newsapi-python")
#         return []
#     except Exception as e:
#         print(f"Error fetching from NewsAPI: {str(e)}")
#         return []


def extract_full_text(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract full text for each article using newspaper3k.
    
    Args:
        articles: List of article dictionaries with URLs
        
    Returns:
        Updated list of article dictionaries with full text
    """
    # Import newspaper3k if needed
    try:
        from newspaper import Article
    except ImportError:
        print("Warning: newspaper3k not installed. Skipping full text extraction.")
        return articles
    
    for i, article_data in enumerate(articles):
        url = article_data["url"]
        if not url:
            continue
            
        print(f"Extracting full text from {url}")
        
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Update the article data with full text
            articles[i]["text"] = article.text
            
            # Add additional metadata if not already present
            if not article_data.get("date"):
                articles[i]["date"] = article.publish_date.isoformat() if article.publish_date else ""
                
            # Avoid hitting rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error extracting text from {url}: {str(e)}")
            
    return articles


def save_articles(articles: List[Dict[str, Any]], ticker: str) -> None:
    """
    Save articles to JSON files.
    
    Args:
        articles: List of article dictionaries
        ticker: Stock ticker symbol
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = DATA_DIR / f"{ticker}_articles_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(articles, f, indent=2)
        
    print(f"Saved {len(articles)} articles to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Scrape news articles for a given stock ticker")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g., TSLA)")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of articles to retrieve per source")
    parser.add_argument("--skip-extract", action="store_true", help="Skip full text extraction")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    # Collect articles from different sources
    yahoo_articles = get_yahoo_finance_news(ticker, args.limit)
    newsapi_articles = []  # get_newsapi_articles(ticker, args.limit)
    
    # Combine articles from all sources
    all_articles = yahoo_articles + newsapi_articles
    
    # Extract full text for articles (if not skipped)
    if not args.skip_extract:
        all_articles = extract_full_text(all_articles)
    
    # Save articles
    save_articles(all_articles, ticker)


if __name__ == "__main__":
    main() 