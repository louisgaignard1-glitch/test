
import numpy as np
import pandas as pd
from scipy.stats import t
import streamlit as st

@st.cache_data
def monte_carlo_simulation(mu, Sigma, allocation, n_simulations=400, n_days=252):

    weights = allocation['Poids'].reindex(mu.index).fillna(0).values

    # regimes
    mu_daily = mu / 252
    Sigma_daily = Sigma / 252

    mu_bull = mu_daily * 1.2
    mu_bear = mu_daily * 0.4

    Sigma_bull = Sigma_daily * 0.8
    Sigma_bear = Sigma_daily * 1.8

    # transition matrix
    P = np.array([[0.95, 0.05],
                  [0.10, 0.90]])

    simulations = np.zeros((n_days, n_simulations))

    for s in range(n_simulations):

        regime = 0
        for t_ in range(n_days):

            if regime == 0:
                shock = t.rvs(df=5, size=len(mu))
                r = mu_bull + np.linalg.cholesky(Sigma_bull) @ shock
            else:
                shock = t.rvs(df=4, size=len(mu))
                r = mu_bear + np.linalg.cholesky(Sigma_bear) @ shock

            simulations[t_, s] = np.dot(r, weights)

            regime = np.random.choice([0,1], p=P[regime])

    cumulative = (1 + simulations).cumprod(axis=0)
    df = pd.DataFrame(cumulative)

    percentiles = {
        "p5": df.quantile(0.05, axis=1),
        "p25": df.quantile(0.25, axis=1),
        "p50": df.quantile(0.50, axis=1),
        "p75": df.quantile(0.75, axis=1),
        "p95": df.quantile(0.95, axis=1),
    }

    final_returns = df.iloc[-1] - 1

    var_95 = np.percentile(final_returns, 5)
    cvar_95 = final_returns[final_returns <= var_95].mean()

    return percentiles, var_95, cvar_95
