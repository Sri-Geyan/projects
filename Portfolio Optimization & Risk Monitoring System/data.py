import yfinance as yf
import pandas as pd

def get_price_data(tickers, start="2019-01-01"):
    data = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False
    )["Close"]

    return data.dropna()

def compute_returns(price_data):
    return price_data.pct_change().dropna()
