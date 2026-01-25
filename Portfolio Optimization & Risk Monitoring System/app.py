import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from data import get_price_data, compute_returns
from optimization import optimize_portfolio_sector_constrained
from risk import value_at_risk, conditional_var, max_drawdown, sector_exposure
from simulation import monte_carlo_simulation

st.set_page_config("Indian Portfolio Optimizer", layout="wide")
st.title("🇮🇳 Indian Portfolio Optimization & Risk Monitoring System")

INDIAN_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "ITC.NS", "LT.NS", "SBIN.NS",
    "HINDUNILVR.NS", "AXISBANK.NS"
]

tickers = st.multiselect(
    "Select NSE Stocks",
    INDIAN_STOCKS,
    default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ITC.NS"]
)

if len(tickers) < 3:
    st.warning("Select at least 3 stocks.")
    st.stop()

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

st.subheader("📊 Portfolio Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Return", f"{annual_return:.2%}")
c2.metric("Volatility", f"{annual_vol:.2%}")
c3.metric("95% VaR", f"{var_95:.2%}")
c4.metric("95% CVaR", f"{cvar_95:.2%}")

# Allocation
st.subheader("📌 Asset Allocation")
for t, w in zip(tickers, weights):
    st.write(f"{t.replace('.NS','')}: {w:.2%}")

# Sector Allocation
st.subheader("🏭 Sector Allocation")
sector_alloc = sector_exposure(weights, tickers)
for sector, alloc in sector_alloc.items():
    st.progress(min(alloc, 1.0))
    st.write(f"{sector}: {alloc:.2%}")

# Monte Carlo
st.subheader("🔮 Monte Carlo Stress Test")
simulated = monte_carlo_simulation(mean_returns, cov_matrix, weights)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(simulated[:, :200], alpha=0.12)
ax.set_title("Simulated Portfolio Growth (INR)")
ax.set_ylabel("Portfolio Value")
st.pyplot(fig)
