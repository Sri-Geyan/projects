from __future__ import annotations
import pandas as pd


def prepare_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute day-over-day and next-day percentage returns from Close prices."""
    df = prices_df.copy()
    df["return"] = df["Close"].pct_change()
    df["next_day_return"] = df["return"].shift(-1)
    return df


def merge_sentiment_prices(
    sentiment_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join daily sentiment scores with price data on date."""
    return pd.merge(
        sentiment_df,
        prices_df,
        left_on="date",
        right_on="Date",
        how="inner",
    )


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between sentiment score and next-day return."""
    return df[["sentiment_score", "next_day_return"]].corr()
