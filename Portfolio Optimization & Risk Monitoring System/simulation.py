import numpy as np

def monte_carlo_simulation(mean_returns, cov_matrix, weights, days=252, simulations=5000):
    mu = np.sum(mean_returns * weights)
    sigma = np.sqrt(weights.T @ cov_matrix @ weights)

    daily_returns = np.random.normal(mu, sigma, (days, simulations))
    return np.cumprod(1 + daily_returns, axis=0)
