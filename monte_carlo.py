import numpy as np
import pandas as pd
from scipy.stats import t
import streamlit as st

@st.cache_data
def monte_carlo_simulation(mu, Sigma, allocation, n_simulations=400, n_days=252):
    weights = allocation['Poids'].reindex(mu.index).fillna(0).values
    mu_daily = mu / 252
    Sigma_daily = Sigma / 252

    simulations = np.zeros((n_days, n_simulations))

    for s in range(n_simulations):
        shock = np.random.multivariate_normal(mu_daily, Sigma_daily, n_days)
        simulations[:, s] = shock @ weights

    cumulative = (1 + simulations).cumprod(axis=0)
    df = pd.DataFrame(cumulative)

    percentiles = {p: df.quantile(q, axis=1) for p, q in zip(["p5","p25","p50","p75","p95"], [0.05,0.25,0.5,0.75,0.95])}

    final_returns = df.iloc[-1] - 1
    var_95 = np.percentile(final_returns, 5)
    cvar_95 = final_returns[final_returns <= var_95].mean()

    return percentiles, var_95, cvar_95
