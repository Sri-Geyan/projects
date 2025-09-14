"""
Stock / Mutual Funds / ETFs Recommendation System (Streamlit UI)
Data Source: Yahoo Finance (yfinance)
Features:
- User enters tickers manually (no defaults)
- Computes risk/return metrics
- Ranks stocks using weighted multi-metric score
- Interactive weight tuning
- Visualizations: scatterplot + bar chart
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# ---------------------------
# Helper: Data collection
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_price_data(tickers, period="3y", interval="1d"):
    """Download adjusted close prices for tickers using yfinance."""
    if isinstance(tickers, str):
        tickers = [tickers]
    ticker_str = " ".join(tickers)
    df = yf.download(ticker_str, period=period, interval=interval, progress=False, threads=True)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.levels[0]:
            price_df = df["Adj Close"].copy()
        elif "Close" in df.columns.levels[0]:
            price_df = df["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        if "Adj Close" in df.columns:
            price_df = df[["Adj Close"]].copy()
            price_df.columns = tickers
        elif "Close" in df.columns:
            price_df = df[["Close"]].copy()
            price_df.columns = tickers
        else:
            return pd.DataFrame()

    price_df = price_df.dropna(how="all")
    return price_df

# ---------------------------
# Feature engineering
# ---------------------------
def compute_features(price_df, risk_free_rate=0.02):
    """Compute return, volatility, Sharpe, drawdown, mean daily return."""
    returns = price_df.pct_change().dropna()
    trading_days = 252.0
    features = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if len(r) < 10:
            continue
        mean_daily = r.mean()
        ann_return = (1 + mean_daily) ** trading_days - 1
        vol = r.std() * np.sqrt(trading_days)
        sharpe = (ann_return - risk_free_rate) / (vol + 1e-9)
        price = price_df[ticker].dropna()
        cum = price / price.cummax()
        max_dd = (cum.min() - 1)
        features.append({
            "ticker": ticker,
            "ann_return": ann_return,
            "volatility": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "mean_daily": mean_daily
        })
    features_df = pd.DataFrame(features).set_index("ticker")
    return features_df

# ---------------------------
# Risk bucketing (K-Means)
# ---------------------------
def risk_bucketing(features_df, n_clusters=3, random_state=42):
    X = features_df[["ann_return", "volatility"]].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(Xs)
    features_df = features_df.copy()
    features_df["risk_bucket"] = labels
    bucket_vols = features_df.groupby("risk_bucket")["volatility"].mean().sort_values()
    mapping = {}
    labels_sorted = list(bucket_vols.index)
    labels_names = ["Low", "Medium", "High"][:len(labels_sorted)]
    for lbl, name in zip(labels_sorted, labels_names):
        mapping[lbl] = name
    features_df["risk_label"] = features_df["risk_bucket"].map(mapping)
    return features_df, kmeans, scaler

# ---------------------------
# Content-based recommendation
# ---------------------------
def build_content_similarity(features_df, feature_cols=None):
    if feature_cols is None:
        feature_cols = ["ann_return", "volatility", "sharpe", "max_drawdown"]
    df = features_df.copy().fillna(0)
    X = df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    sim = cosine_similarity(Xs)
    sim_df = pd.DataFrame(sim, index=df.index, columns=df.index)
    return sim_df, scaler

def recommend_content(ticker, sim_df, top_n=10, exclude_self=True):
    if ticker not in sim_df.index:
        return []
    series = sim_df[ticker].sort_values(ascending=False)
    if exclude_self:
        series = series.drop(labels=[ticker])
    return list(series.head(top_n).index)

# ---------------------------
# Collaborative boost (synthetic demo)
# ---------------------------
def build_synthetic_user_matrix(tickers, n_users=200, random_state=42):
    rng = np.random.default_rng(random_state)
    base_popularity = np.linspace(1, 0.1, num=len(tickers))
    matrix = (rng.random((n_users, len(tickers))) < base_popularity).astype(float)
    for u in range(n_users):
        if u % 10 == 0:
            matrix[u, :5] = (rng.random(5) < 0.6).astype(float)
    return pd.DataFrame(matrix, columns=tickers)

def collaborative_score_for_tickers(user_item_df):
    svd = TruncatedSVD(n_components=10, random_state=0)
    item_matrix = user_item_df.T.fillna(0).values
    item_factors = svd.fit_transform(item_matrix)
    return pd.DataFrame(item_factors, index=user_item_df.columns)

# ---------------------------
# Portfolio optimizer
# ---------------------------
def portfolio_optimization(returns_df, selected_tickers, risk_free_rate=0.02):
    R = returns_df[selected_tickers].dropna()
    mu = R.mean() * 252
    cov = R.cov() * 252
    n = len(selected_tickers)
    x0 = np.ones(n) / n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n))

    def neg_sharpe(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        sharpe = (port_return - risk_free_rate) / (port_vol + 1e-9)
        return -sharpe

    result = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else x0
    weights_series = pd.Series(weights, index=selected_tickers)
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    port_sharpe = (port_return - risk_free_rate) / (port_vol + 1e-9)
    stats = {"return": port_return, "volatility": port_vol, "sharpe": port_sharpe}
    return weights_series, stats

# ---------------------------
# Ranking function
# ---------------------------
def rank_stocks(features_df, weights=None):
    df = features_df.copy()
    if weights is None:
        weights = {
            "ann_return": 0.25,
            "volatility": 0.15,
            "sharpe": 0.30,
            "max_drawdown": 0.15,
            "mean_daily": 0.15
        }

    def normalize(series, inverse=False):
        if series.nunique() == 1:
            return pd.Series([0.5]*len(series), index=series.index)
        norm = (series - series.min()) / (series.max() - series.min())
        return 1 - norm if inverse else norm

    df["ann_return_n"]   = normalize(df["ann_return"])
    df["volatility_n"]   = normalize(df["volatility"], inverse=True)
    df["sharpe_n"]       = normalize(df["sharpe"])
    df["max_drawdown_n"] = normalize(df["max_drawdown"], inverse=True)
    df["mean_daily_n"]   = normalize(df["mean_daily"])

    df["score"] = (
        df["ann_return_n"]   * weights["ann_return"] +
        df["volatility_n"]   * weights["volatility"] +
        df["sharpe_n"]       * weights["sharpe"] +
        df["max_drawdown_n"] * weights["max_drawdown"] +
        df["mean_daily_n"]   * weights["mean_daily"]
    )
    return df.sort_values("score", ascending=False)

# ---------------------------
# Utilities
# ---------------------------
def plot_risk_return_scatter(features_df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=features_df.reset_index(),
                    x="volatility", y="ann_return", hue="risk_label", s=80, ax=ax)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Risk vs Return")
    plt.tight_layout()
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Investment Recommendation System", layout="wide")
st.title("📈 Investment Recommendation System — Stocks / ETFs / Mutual Funds (Yahoo Finance)")

with st.sidebar:
    st.header("Configuration")
    tickers_input = st.text_area("Enter tickers (comma separated)", value="", height=120, placeholder="e.g., AAPL, MSFT, TSLA, SPY")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    period = st.selectbox("Price history period", options=["1y","2y","3y","5y"], index=2)
    interval = st.selectbox("Interval", options=["1d","1wk","1mo"], index=0)
    top_n = st.slider("Number of recommendations", 3, 20, 8)
    use_collab = st.checkbox("Use collaborative boost (synthetic demo)", value=True)

    st.markdown("### ⚖️ Ranking Weights")
    w_ann_return   = st.slider("Weight: Annual Return", 0.0, 1.0, 0.25, 0.05)
    w_volatility   = st.slider("Weight: Volatility", 0.0, 1.0, 0.15, 0.05)
    w_sharpe       = st.slider("Weight: Sharpe Ratio", 0.0, 1.0, 0.30, 0.05)
    w_max_dd       = st.slider("Weight: Max Drawdown", 0.0, 1.0, 0.15, 0.05)
    w_mean_daily   = st.slider("Weight: Mean Daily Return", 0.0, 1.0, 0.15, 0.05)

    total = w_ann_return + w_volatility + w_sharpe + w_max_dd + w_mean_daily
    if total == 0: total = 1.0
    weights = {
        "ann_return": w_ann_return / total,
        "volatility": w_volatility / total,
        "sharpe": w_sharpe / total,
        "max_drawdown": w_max_dd / total,
        "mean_daily": w_mean_daily / total
    }

# Validate user input
if not tickers:
    st.error("Please enter at least one valid ticker symbol to continue.")
    st.stop()

with st.spinner("Fetching price data..."):
    price_df = fetch_price_data(tickers, period=period, interval=interval)
    if price_df.empty:
        st.error("No price data fetched. Check tickers.")
        st.stop()

with st.spinner("Computing features..."):
    features_df = compute_features(price_df)

features_df, _, _ = risk_bucketing(features_df, n_clusters=3)
sim_df, _ = build_content_similarity(features_df)

collab_factors = None
if use_collab:
    user_item_df = build_synthetic_user_matrix(list(features_df.index), n_users=300)
    collab_factors = collaborative_score_for_tickers(user_item_df)

# Asset Features Table
st.subheader("Asset Features")
st.dataframe(features_df.style.format({
    "ann_return": "{:.2%}",
    "volatility": "{:.2%}",
    "sharpe": "{:.2f}",
    "max_drawdown": "{:.2%}",
    "mean_daily": "{:.4f}"
}))

# Risk vs Return Plot
st.subheader("Risk vs Return")
st.pyplot(plot_risk_return_scatter(features_df))

# Recommendations
st.subheader("Recommendations")
seed_ticker = st.selectbox("Pick a seed asset", options=list(features_df.index))
candidates = recommend_content(seed_ticker, sim_df, top_n=top_n*3)
ranked_recs = features_df.loc[candidates].sort_values("sharpe", ascending=False).head(top_n)
st.table(ranked_recs[["ann_return","volatility","sharpe","max_drawdown","mean_daily","risk_label"]].style.format({
    "ann_return": "{:.2%}", "volatility": "{:.2%}", "sharpe": "{:.2f}", "max_drawdown": "{:.2%}", "mean_daily": "{:.4f}"
}))

# Stock Ranking Section
st.subheader("📊 Stock Ranking (Best → Worst)")
ranked_df = rank_stocks(features_df, weights=weights)
st.dataframe(ranked_df[[
    "ann_return","volatility","sharpe","max_drawdown","mean_daily","score","risk_label"
]].style.format({
    "ann_return": "{:.2%}", "volatility": "{:.2%}", "sharpe": "{:.2f}",
    "max_drawdown": "{:.2%}", "mean_daily": "{:.4f}", "score": "{:.3f}"
}))

# Bar chart of Top 10 Ranked Stocks
st.subheader("🏆 Top 10 Ranked Stocks (by Score)")
top10 = ranked_df.head(10).reset_index()
chart = alt.Chart(top10).mark_bar().encode(
    x=alt.X("score:Q", title="Composite Score"),
    y=alt.Y("ticker:N", sort="-x", title="Ticker"),
    color=alt.Color("risk_label:N", legend=alt.Legend(title="Risk Level")),
    tooltip=["ticker", "ann_return", "volatility", "sharpe", "max_drawdown", "mean_daily", "score"]
).properties(width=700, height=400)
st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption("This demo is educational — not financial advice. Use reliable APIs and validate results before making investment decisions.")
