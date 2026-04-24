import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data import get_price_data, compute_returns
from optimization import optimize_portfolio_sector_constrained, optimize_min_volatility, optimize_equal_weight
from risk import value_at_risk, conditional_var, max_drawdown, sector_exposure
from simulation import monte_carlo_simulation

# Premium CSS
st.set_page_config(page_title="Indian Portfolio Optimizer", layout="wide", page_icon="🇮🇳")

# ... (CSS remains same)

st.markdown('<h1 class="gradient-text">🇮🇳 Indian Portfolio Optimization & Risk Monitoring System</h1>', unsafe_allow_html=True)

INDIAN_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS", "LT.NS", "SBIN.NS",
    "HINDUNILVR.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "HCLTECH.NS", "TITAN.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "ULTRACEMCO.NS",
    "NTPC.NS", "POWERGRID.NS", "JSWSTEEL.NS", "M&M.NS", "ADANIENT.NS", "TATASTEEL.NS",
    "BAJAJFINSV.NS", "INDUSINDBK.NS", "ADANIPORTS.NS", "GRASIM.NS", "NESTLEIND.NS", "ONGC.NS",
    "TECHM.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "COALINDIA.NS", "TATACONSUM.NS", "CIPLA.NS",
    "APOLLOHOSP.NS", "DRREDDY.NS", "SBILIFE.NS", "BPCL.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "DIVISLAB.NS", "HDFCLIFE.NS", "LTIM.NS", "UPL.NS", "WIPRO.NS", "HEROMOTOCO.NS"
]

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers = st.multiselect(
        "Select NSE Stocks (Max 50)",
        INDIAN_STOCKS,
        default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ITC.NS", "MARUTI.NS"]
    )
    
    strategy = st.radio(
        "Optimization Strategy",
        ["Sector-Constrained Sharpe", "Minimum Volatility", "Equal Weight"]
    )
    
    if len(tickers) < 3:
        st.warning("Please select at least 3 stocks to optimize.")
        st.stop()

with st.spinner("Fetching Market Data & Optimizing Portfolio..."):
    prices = get_price_data(tickers)
    returns = compute_returns(prices)

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    if strategy == "Sector-Constrained Sharpe":
        weights = optimize_portfolio_sector_constrained(mean_returns, cov_matrix, tickers)
    elif strategy == "Minimum Volatility":
        weights = optimize_portfolio_sector_constrained(mean_returns, cov_matrix, tickers) # Fallback to constraints if possible, or just min vol
        weights = optimize_min_volatility(mean_returns, cov_matrix, tickers)
    else:
        weights = optimize_equal_weight(mean_returns, cov_matrix, tickers)
        
    portfolio_returns = returns @ weights

    # Metrics
    annual_return = np.sum(mean_returns * weights) * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    var_95 = value_at_risk(portfolio_returns)
    cvar_95 = conditional_var(portfolio_returns)
    mdd = max_drawdown((1 + portfolio_returns).cumprod())

# ... (Metrics display remains same)

st.subheader("🔮 Monte Carlo Stress Test (10,000 Simulations)")
st.info("Simulating future portfolio growth paths over 252 trading days. Showing 100 sample paths.")

with st.spinner("Running Monte Carlo..."):
    simulated = monte_carlo_simulation(mean_returns, cov_matrix, weights, days=252, simulations=10000)
    
    plot_samples = simulated[:, :100]
    
    # ... (Plotting remains same)
