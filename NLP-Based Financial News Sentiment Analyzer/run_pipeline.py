"""
run_pipeline.py — Batch NLP + Correlation Pipeline

Executes end-to-end:
  1. Load news headlines and price data
  2. Preprocess text
  3. Score with FinBERT
  4. Aggregate daily sentiment
  5. Compute next-day return correlation
"""

import logging
import sys

import pandas as pd

from src.load_data import load_news, load_prices
from src.preprocess import clean_text
from src.sentiment import get_sentiment
from src.analysis import prepare_returns, merge_sentiment_prices, compute_correlation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("=== Financial News Sentiment Pipeline ===")

    # 1. Load data
    log.info("Loading news data…")
    try:
        news = load_news()
    except FileNotFoundError as e:
        log.error(f"Could not find news CSV: {e}")
        sys.exit(1)

    log.info("Loading price data…")
    try:
        prices = load_prices()
    except FileNotFoundError as e:
        log.error(f"Could not find prices CSV: {e}")
        sys.exit(1)

    log.info(f"  {len(news)} headlines  |  {len(prices)} price rows")

    # 2. Preprocess
    log.info("Preprocessing headlines…")
    news["clean_headline"] = news["headline"].apply(clean_text)

    # 3. FinBERT scoring (this triggers model download on first run ~400 MB)
    log.info("Running FinBERT sentiment inference…  (first run downloads ~400 MB)")
    news["sentiment_score"] = news["clean_headline"].apply(get_sentiment)
    log.info("  Scoring complete.")

    # 4. Daily aggregation
    daily_sentiment = (
        news.groupby(["date", "ticker"])["sentiment_score"]
        .mean()
        .reset_index()
    )
    log.info(f"  {len(daily_sentiment)} daily sentiment rows")

    # 5. Returns & correlation
    prices = prepare_returns(prices)
    final_df = merge_sentiment_prices(daily_sentiment, prices)

    if final_df.empty:
        log.warning("No overlapping dates between sentiment and price data — check CSVs.")
        return

    corr = compute_correlation(final_df)
    print("\n=== Correlation Matrix ===")
    print(corr.to_string())
    print(f"\nSentiment → Next-Day Return Pearson r: "
          f"{corr.loc['sentiment_score','next_day_return']:.4f}")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
