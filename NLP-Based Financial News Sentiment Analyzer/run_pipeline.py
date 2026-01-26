from src.load_data import load_news, load_prices
from src.preprocess import clean_text
from src.sentiment import get_sentiment
from src.analysis import prepare_returns, merge_sentiment_prices, compute_correlation

print("🚀 run_pipeline.py started")


def main():
    print("Loading data...")
    news = load_news()
    prices = load_prices()

    print("Preprocessing news...")
    news["clean_headline"] = news["headline"].apply(clean_text)

    print("Running FinBERT sentiment...")
    news["sentiment_score"] = news["clean_headline"].apply(get_sentiment)

    daily_sentiment = (
        news.groupby(["date", "ticker"])["sentiment_score"]
        .mean()
        .reset_index()
    )

    prices = prepare_returns(prices)

    final_df = merge_sentiment_prices(daily_sentiment, prices)

    print("\nCorrelation Matrix:")
    print(compute_correlation(final_df))

if __name__ == "__main__":
    main()
