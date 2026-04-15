from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "data"


def load_news(path: Optional[str] = None) -> pd.DataFrame:
    """Load financial news headlines. Defaults to data/news.csv relative to project root."""
    fpath = Path(path) if path else _DATA_DIR / "news.csv"
    df = pd.read_csv(fpath)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_prices(path: Optional[str] = None) -> pd.DataFrame:
    """Load OHLC stock prices. Defaults to data/prices.csv relative to project root."""
    fpath = Path(path) if path else _DATA_DIR / "prices.csv"
    df = pd.read_csv(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
