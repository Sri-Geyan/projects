import yfinance as yf
import pandas as pd
import ssl
import urllib.request

# macOS SSL Patch
try:
    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx)))
except Exception:
    pass

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
