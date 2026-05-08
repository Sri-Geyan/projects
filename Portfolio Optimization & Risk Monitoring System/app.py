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


# ... (CSS remains same)

st.markdown('<h1 class="gradient-text">🇮🇳 Indian Portfolio Optimization & Risk Monitoring System</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers_input = st.text_input(
        "Enter Stock Tickers (comma-separated)",
        ""
    )
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    
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

    # Metrics
    st.subheader("📊 Portfolio Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Expected Return", f"{annual_return*100:.2f}%")
    col2.metric("Annual Volatility", f"{annual_vol*100:.2f}%")
    col3.metric("VaR (95%)", f"{var_95*100:.2f}%")
    col4.metric("CVaR (95%)", f"{cvar_95*100:.2f}%")
    col5.metric("Max Drawdown", f"{mdd*100:.2f}%")
    
    st.subheader("⚖️ Optimized Weights")
    weight_fig = px.pie(names=tickers, values=weights, hole=0.3)
    st.plotly_chart(weight_fig, use_container_width=True)
st.subheader("🔮 Monte Carlo Stress Test (10,000 Simulations)")
st.info("Simulating future portfolio growth paths over 252 trading days. Showing 100 sample paths.")

with st.spinner("Running Monte Carlo..."):
    simulated = monte_carlo_simulation(mean_returns, cov_matrix, weights, days=252, simulations=10000)
    
    plot_samples = simulated[:, :100]
    
    fig = go.Figure()
    for i in range(100):
        fig.add_trace(go.Scatter(y=plot_samples[:, i], mode='lines', line=dict(color='rgba(0,100,255,0.1)'), showlegend=False))
    
    fig.update_layout(title="Monte Carlo Portfolio Growth Paths", xaxis_title="Days", yaxis_title="Portfolio Value")
    st.plotly_chart(fig, use_container_width=True)
