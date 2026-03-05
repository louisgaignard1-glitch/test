import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def monte_carlo_simulation(mu, Sigma, allocation, n_simulations=400, n_days=252):
    # Convertir mu en tableau NumPy
    mu_daily = mu.values / 252
    Sigma_daily = Sigma.values / 252

    # Ajouter une petite valeur à la diagonale pour éviter les problèmes numériques
    Sigma_daily += 1e-6 * np.eye(len(Sigma_daily))

    # Décomposition de Cholesky
    try:
        L = np.linalg.cholesky(Sigma_daily)
    except np.linalg.LinAlgError:
        st.error("La matrice de covariance n'est pas inversible. Vérifiez les données.")
        return {}, np.nan, np.nan

    # Initialiser les simulations
    simulations = np.zeros((n_days, n_simulations))
    weights = allocation['Poids'].reindex(mu.index).fillna(0).values

    for s in range(n_simulations):
        z = np.random.normal(size=n_days * len(mu_daily)).reshape(n_days, len(mu_daily))
        shock = mu_daily + z @ L.T
        simulations[:, s] = shock @ weights

    cumulative = (1 + simulations).cumprod(axis=0)
    df = pd.DataFrame(cumulative)

    percentiles = {
        p: df.quantile(q, axis=1)
        for p, q in zip(["p5", "p25", "p50", "p75", "p95"], [0.05, 0.25, 0.5, 0.75, 0.95])
    }

    final_returns = df.iloc[-1] - 1
    var_95 = np.percentile(final_returns, 5)
    cvar_95 = final_returns[final_returns <= var_95].mean()

    return percentiles, var_95, cvar_95
