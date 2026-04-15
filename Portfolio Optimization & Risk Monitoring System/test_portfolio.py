import pytest
import numpy as np
import pandas as pd
from data import compute_returns
from risk import value_at_risk, conditional_var, max_drawdown, sector_exposure
from simulation import monte_carlo_simulation

@pytest.fixture
def dummy_price_data():
    dates = pd.date_range("2023-01-01", periods=5)
    data = {
        "RELIANCE.NS": [2000, 2020, 2010, 2050, 2100],
        "TCS.NS": [3000, 2950, 3050, 3100, 3120]
    }
    return pd.DataFrame(data, index=dates)

def test_compute_returns(dummy_price_data):
    returns = compute_returns(dummy_price_data)
    assert returns.shape == (4, 2)
    assert not returns.isnull().values.any()
    # Test specific calculation
    assert np.isclose(returns.iloc[0]["RELIANCE.NS"], 2020/2000 - 1)

def test_risk_metrics():
    # Deterministic dummy returns
    returns = np.array([-0.05, -0.02, 0.01, 0.03, 0.04, 0.05, -0.06])
    
    var_95 = value_at_risk(returns, confidence=0.95)
    assert var_95 < 0
    
    cvar_95 = conditional_var(returns, confidence=0.95)
    assert cvar_95 <= var_95
    
    # Cumulative returns for mdd
    cum_ret = np.cumprod(1 + returns)
    mdd = max_drawdown(cum_ret)
    assert mdd < 0

def test_sector_exposure():
    tickers = ["HDFCBANK.NS", "TCS.NS", "ITC.NS"]
    weights = [0.4, 0.3, 0.3]
    exposure = sector_exposure(weights, tickers)
    
    assert exposure["BANKING"] == 0.4
    assert exposure["IT"] == 0.3
    assert exposure["FMCG"] == 0.3

def test_monte_carlo_simulation():
    mean_returns = pd.Series([0.001, 0.002])
    cov_matrix = pd.DataFrame([
        [0.0001, 0.00005],
        [0.00005, 0.0002]
    ])
    weights = np.array([0.5, 0.5])
    
    # 10 days, 5 sims
    sim = monte_carlo_simulation(mean_returns, cov_matrix, weights, days=10, simulations=5)
    assert sim.shape == (10, 5)
