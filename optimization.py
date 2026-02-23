import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_portfolio(mu, Sigma, assets):
    n = len(assets)

    def port_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))

    initial_weights = np.ones(n) / n
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))

    result_min_var = minimize(port_vol, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    allocation = pd.DataFrame({'Poids': result_min_var.x}, index=assets)
    return result_min_var, allocation

def calculate_sharpe_ratio(weights, mu, Sigma, risk_free_rate=0.02):
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
    return float((port_return - risk_free_rate) / port_vol)

def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return float(drawdown.min())
