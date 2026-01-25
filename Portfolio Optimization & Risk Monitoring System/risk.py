import numpy as np
from sectors import SECTOR_MAP

def value_at_risk(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)

def conditional_var(returns, confidence=0.95):
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()

def max_drawdown(cumulative_returns):
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def sector_exposure(weights, tickers):
    exposure = {}

    for sector, stocks in SECTOR_MAP.items():
        idx = [i for i, t in enumerate(tickers) if t in stocks]
        exposure[sector] = round(sum(weights[i] for i in idx), 4)

    return exposure
