#!/bin/bash
# Script to gather a massive dataset of financial news articles

# Create output directory
mkdir -p data

# Set variables
OUTPUT_BASE="data/financial_news_massive"
DATE=$(date +"%Y%m%d")
OUTPUT="${OUTPUT_BASE}_${DATE}"

echo "========================================"
echo "Starting massive data collection process"
echo "========================================"
echo "Output files will be saved to: ${OUTPUT}_*.json"
echo "Starting at: $(date)"
echo "----------------------------------------"

# Run the gatherer with optimal settings
python gather_training_data.py \
  --use-all-tickers \
  --articles-per-ticker 100 \
  --max-workers 3 \
  --batch-size 10 \
  --delay 3 \
  --output "${OUTPUT}.json" \
  --start-date "2024-01-01" \
  --end-date "2024-12-31"

echo "----------------------------------------"
echo "Massive data gathering completed"
echo "Ended at: $(date)"
echo "========================================"

# Count the articles
TRAIN_COUNT=$(python -c "import json; print(len(json.load(open('${OUTPUT}_train.json'))))")
EVAL_COUNT=$(python -c "import json; print(len(json.load(open('${OUTPUT}_eval.json'))))")
TOTAL=$((TRAIN_COUNT + EVAL_COUNT))

echo "Collected a total of ${TOTAL} articles"
echo "  - Training set: ${TRAIN_COUNT} articles"
echo "  - Evaluation set: ${EVAL_COUNT} articles"
echo "========================================"

# Generate report of tickers and counts
echo "Generating ticker distribution report..."
python -c "
import json
from collections import Counter

# Load data
train = json.load(open('${OUTPUT}_train.json'))
eval_data = json.load(open('${OUTPUT}_eval.json'))
all_data = train + eval_data

# Count articles per ticker
ticker_counts = Counter([article.get('ticker', 'UNKNOWN') for article in all_data])
total = len(all_data)

# Write report
with open('${OUTPUT}_report.txt', 'w') as f:
    f.write(f'Ticker distribution for {total} articles:\\n')
    f.write('=' * 40 + '\\n')
    f.write(f'{"Ticker":<10} {"Count":<8} {"Percentage":<10}\\n')
    f.write('-' * 40 + '\\n')
    
    for ticker, count in ticker_counts.most_common():
        pct = count / total * 100
        f.write(f'{ticker:<10} {count:<8} {pct:.2f}%\\n')
"

echo "Report saved to ${OUTPUT}_report.txt"
echo "========================================"

# Optional: Create a compressed archive for easy sharing
echo "Creating compressed archive of dataset..."
tar -czf "${OUTPUT}.tar.gz" "${OUTPUT}_train.json" "${OUTPUT}_eval.json" "${OUTPUT}_report.txt"
echo "Archive saved to ${OUTPUT}.tar.gz"
echo "========================================"

echo "All done!" 