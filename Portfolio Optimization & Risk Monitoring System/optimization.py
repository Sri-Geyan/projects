import numpy as np
from scipy.optimize import minimize
from sectors import SECTOR_MAP, SECTOR_LIMITS

INDIA_RISK_FREE_RATE = 0.072  # RBI 10Y G-Sec

def optimize_portfolio_sector_constrained(mean_returns, cov_matrix, tickers):
    n = len(mean_returns)

    def negative_sharpe(weights):
        annual_return = np.sum(mean_returns * weights) * 252
        annual_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
        return -(annual_return - INDIA_RISK_FREE_RATE) / annual_vol

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    for sector, stocks in SECTOR_MAP.items():
        indices = [i for i, t in enumerate(tickers) if t in stocks]
        if not indices:
            continue

        min_w, max_w = SECTOR_LIMITS[sector]

        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=indices, m=min_w: np.sum(w[idx]) - m
        })

        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=indices, m=max_w: m - np.sum(w[idx])
        })

    bounds = tuple((0, 0.30) for _ in range(n))
    initial = np.array(n * [1 / n])

    result = minimize(
        negative_sharpe,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x
