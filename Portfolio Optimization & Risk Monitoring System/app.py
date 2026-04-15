import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data import get_price_data, compute_returns
from optimization import optimize_portfolio_sector_constrained
from risk import value_at_risk, conditional_var, max_drawdown, sector_exposure
from simulation import monte_carlo_simulation

# Premium CSS
st.set_page_config(page_title="Indian Portfolio Optimizer", layout="wide", page_icon="🇮🇳")

st.markdown("""
<style>
/* Modern Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Gradient text */
.gradient-text {
    background: -webkit-linear-gradient(45deg, #FF9933, #FFFFFF, #138808);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 2rem;
}

.stMetric {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="gradient-text">🇮🇳 Indian Portfolio Optimization & Risk Monitoring System</h1>', unsafe_allow_html=True)

INDIAN_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "ITC.NS", "LT.NS", "SBIN.NS",
    "HINDUNILVR.NS", "AXISBANK.NS"
]

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers = st.multiselect(
        "Select NSE Stocks",
        INDIAN_STOCKS,
        default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ITC.NS"]
    )
    if len(tickers) < 3:
        st.warning("Please select at least 3 stocks to optimize.")
        st.stop()
    st.info("The optimization applies limits to prevent sector over-concentration (Banking, IT, FMCG).")

with st.spinner("Fetching Market Data & Optimizing Portfolio..."):
    prices = get_price_data(tickers)
    returns = compute_returns(prices)

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    weights = optimize_portfolio_sector_constrained(mean_returns, cov_matrix, tickers)
    portfolio_returns = returns @ weights

    # Metrics
    annual_return = np.sum(mean_returns * weights) * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    var_95 = value_at_risk(portfolio_returns)
    cvar_95 = conditional_var(portfolio_returns)
    mdd = max_drawdown((1 + portfolio_returns).cumprod())

st.subheader("📊 Portfolio Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Expected Return", f"{annual_return:.2%}")
col2.metric("Annual Volatility", f"{annual_vol:.2%}")
col3.metric("95% VaR", f"{var_95:.2%}")
col4.metric("95% CVaR", f"{cvar_95:.2%}")
col5.metric("Max Drawdown", f"{mdd:.2%}")

st.markdown("---")

col_alloc, col_sector = st.columns(2)

with col_alloc:
    st.subheader("📌 Optimized Asset Allocation")
    # Donut chart for allocation
    labels = [t.replace('.NS','') for t in tickers]
    fig_alloc = go.Figure(data=[go.Pie(labels=labels, values=weights, hole=.4, hoverinfo='label+percent')])
    fig_alloc.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_alloc, use_container_width=True)

with col_sector:
    st.subheader("🏭 Sector Exposure")
    sector_alloc = sector_exposure(weights, tickers)
    
    sectors_names = list(sector_alloc.keys())
    sectors_vals = list(sector_alloc.values())
    
    fig_sec = px.bar(x=sectors_vals, y=sectors_names, orientation='h', color=sectors_names, text=[f"{v:.1%}" for v in sectors_vals])
    fig_sec.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, showlegend=False, xaxis_title="Weight", yaxis_title="Sector", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sec, use_container_width=True)

st.markdown("---")

st.subheader("🔮 Monte Carlo Stress Test (5000 Simulations)")
st.info("Simulating future portfolio growth paths over 252 trading days. Showing 100 sample paths.")

with st.spinner("Running Monte Carlo..."):
    # Limiting visual traces to 100 so browser doesn't lag too hard
    simulated = monte_carlo_simulation(mean_returns, cov_matrix, weights, days=252, simulations=5000)
    
    plot_samples = simulated[:, :100]
    
    fig_mc = go.Figure()
    for i in range(plot_samples.shape[1]):
        fig_mc.add_trace(go.Scatter(y=plot_samples[:, i], mode='lines', line=dict(color='rgba(0, 150, 255, 0.05)'), showlegend=False))
        
    # Add Mean Path
    mean_path = np.mean(simulated, axis=1)
    fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color='#FF9933', width=3), name='Mean Path'))
    
    # Add 5th Percentile (Worst Case)
    p5_path = np.percentile(simulated, 5, axis=1)
    fig_mc.add_trace(go.Scatter(y=p5_path, mode='lines', line=dict(color='#ff2b2b', width=2, dash='dash'), name='5th Percentile (Tail Risk)'))
    
    fig_mc.update_layout(
        xaxis_title="Trading Days", 
        yaxis_title="Simulated Portfolio Value (1 Unit Initial)",
        height=500,
        margin=dict(t=30, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_mc, use_container_width=True)
