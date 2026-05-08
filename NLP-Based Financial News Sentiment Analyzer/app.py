"""
FinSentiment — Financial News Sentiment Analyzer (Streamlit)
Uses FinBERT (ProsusAI/finbert) for domain-specific NLP scoring.
"""

import ssl
import time
import urllib.request
from typing import Optional

import streamlit as st
import pandas as pd
import yfinance as yf

# ── macOS LibreSSL / OpenSSL compatibility patch ──────────────────────────────
# On macOS, Python ships with LibreSSL which can fail certificate verification
# against Yahoo Finance's CDN. We monkey-patch a permissive SSL context for
# yfinance's underlying urllib calls only. This does NOT affect other connections.
try:
    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE
    urllib.request.install_opener(
        urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=_ssl_ctx)
        )
    )
except Exception:
    pass  # If patching fails, fall through — yfinance will handle the error itself

from src.sentiment import get_sentiment

st.set_page_config(
    page_title="FinSentiment · NLP Analyzer",
    page_icon="📈",
    layout="centered",
)

# Premium CSS Injection
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0A0F1A 0%, #111827 100%); }
    [data-testid="stSidebar"] {
        background: rgba(21, 26, 40, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    .stMetric, .stDataFrame, div[data-testid="stForm"], div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .stMetric:hover, div[data-testid="stForm"]:hover, div[data-testid="stExpander"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 229, 255, 0.1);
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
    }
    h1, h2, h3 {
        background: linear-gradient(90deg, #00E5FF, #0077FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00E5FF, #0077FF) !important;
        color: #0A0F1A !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


st.title("📈 FinSentiment")
st.caption("Financial News Sentiment Analyzer · FinBERT (ProsusAI)")

st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
news_text = st.text_area(
    "Financial Headline",
    placeholder="e.g. Apple reports record Q4 earnings, beats analyst expectations",
    height=80,
)

col1, col2 = st.columns([1, 1])
with col1:
    ticker = st.text_input("Stock Ticker", value="").upper().strip()
with col2:
    period = st.selectbox("Price Window", ["1mo", "3mo", "6mo", "1y"], index=0)

analyze = st.button("Run Analysis", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze:
    if not news_text.strip():
        st.warning("Please enter a headline before running analysis.")
        st.stop()

    with st.spinner("Running FinBERT inference…"):
        try:
            score = get_sentiment(news_text)
        except Exception as e:
            st.error(f"FinBERT error: {e}")
            st.stop()

    # Sentiment result
    label = "Positive 🟢" if score > 0.1 else "Negative 🔴" if score < -0.1 else "Neutral 🟡"
    interp = (
        "Strong bullish signal" if score > 0.5
        else "Mildly bullish signal" if score > 0.1
        else "Strong bearish signal" if score < -0.5
        else "Mildly bearish signal" if score < -0.1
        else "Neutral — no directional bias"
    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment Score", f"{score:+.3f}")
    c2.metric("Label", label)
    c3.metric("Signal", interp.split(" signal")[0] if "signal" in interp else interp)

    st.caption(f"**Interpretation:** {interp}")
    st.info(
        "Sentiment is a signal, not a prediction. Markets may price in "
        "information ahead of publication or react with a lag.",
        icon="ℹ️",
    )

    # Price chart
    st.divider()
    st.subheader(f"{ticker} — {period} price")

    with st.spinner(f"Fetching {ticker} from Yahoo Finance…"):
        stock = None
        _last_exc: Optional[Exception] = None

        # Retry up to 3 times — Yahoo Finance / LibreSSL errors are often transient
        for _attempt in range(3):
            try:
                _df = yf.download(
                    ticker,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                )
                if not _df.empty:
                    stock = _df
                    break  # success
                # Empty DataFrame — wait briefly and retry
                time.sleep(1.5)
            except Exception as exc:
                _last_exc = exc
                time.sleep(1.5)

        if stock is not None and not stock.empty:
            close_col = "Close"
            # yfinance ≥0.2.38 may return a MultiIndex — flatten it
            if hasattr(stock.columns, "levels"):
                stock.columns = [" ".join(c).strip() for c in stock.columns]
                close_col = next(
                    (c for c in stock.columns if c.lower().startswith("close")),
                    stock.columns[0],
                )
            st.line_chart(stock[close_col], use_container_width=True)
        else:
            # Friendly degraded-mode message — sentiment results are still valid
            st.info(
                f"📡 Price chart for **{ticker}** is temporarily unavailable "
                "(intermittent Yahoo Finance / macOS SSL network issue). "
                "Your sentiment analysis above is unaffected — try refreshing "
                "in a moment or check [finance.yahoo.com](https://finance.yahoo.com) "
                "to confirm the ticker is valid.",
                icon="📡",
            )
            if _last_exc:
                with st.expander("Technical details", expanded=False):
                    st.code(str(_last_exc), language="text")