import pandas as pd

def prepare_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    prices_df = prices_df.copy()
    prices_df["return"] = prices_df["Close"].pct_change()
    prices_df["next_day_return"] = prices_df["return"].shift(-1)
    return prices_df

def merge_sentiment_prices(sentiment_df: pd.DataFrame,
                            prices_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        sentiment_df,
        prices_df,
        left_on="date",
        right_on="Date",
        how="inner"
    )

def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    return df[["sentiment_score", "next_day_return"]].corr()
