import numpy as np
import pandas as pd
import streamlit as st


def make_positive_semidefinite(matrix):

    matrix = np.array(matrix)

    matrix = (matrix + matrix.T) / 2

    eigvals, eigvecs = np.linalg.eigh(matrix)

    eigvals[eigvals < 0] = 0

    matrix_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return matrix_psd


@st.cache_data
def monte_carlo_simulation(mu, Sigma, allocation, n_simulations=400, n_days=252):

    weights = allocation['Poids'].reindex(mu.index).fillna(0).values

    mu_daily = mu / 252

    Sigma_daily = Sigma / 252

    Sigma_daily = Sigma_daily.fillna(0)

    Sigma_daily = make_positive_semidefinite(Sigma_daily.values)

    Sigma_daily += np.eye(len(Sigma_daily)) * 1e-10

    simulations = np.zeros((n_days, n_simulations))

    for s in range(n_simulations):

        shock = np.random.multivariate_normal(
            mean=mu_daily.values,
            cov=Sigma_daily,
            size=n_days,
            check_valid='ignore'
        )

        simulations[:, s] = shock @ weights

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
