
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import numpy as np
from scipy.optimize import minimize

def plot_efficient_frontier(result_min_var, mu, Sigma, assets):
    target_returns = np.linspace(mu.min(), mu.max(), 20)
    frontier_vols = []
    frontier_returns = []

    for target_return in target_returns:
        constraints_frontier = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, target_return=target_return: np.dot(w, mu) - target_return}
        )

        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(Sigma, w))),
            x0=np.ones(len(assets)) / len(assets),
            method='SLSQP',
            bounds=tuple((0, 1) for _ in range(len(assets))),
            constraints=constraints_frontier
        )

        if result.success:
            frontier_vols.append(np.sqrt(np.dot(result.x.T, np.dot(Sigma, result.x))))
            frontier_returns.append(target_return)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_vols,
        y=frontier_returns,
        mode='lines+markers',
        name="Efficient frontier"
    ))

    if result_min_var.success:
        fig.add_trace(go.Scatter(
            x=[np.sqrt(np.dot(result_min_var.x.T, np.dot(Sigma, result_min_var.x)))],
            y=[np.dot(result_min_var.x, mu)],
            mode='markers',
            marker=dict(size=12, color='red'),
            name="Minimum variance"
        ))

    for i, asset in enumerate(assets):
        fig.add_trace(go.Scatter(
            x=[np.sqrt(Sigma.iloc[i, i])],
            y=[mu.loc[asset]],
            mode='markers',
            marker=dict(size=8, color='gray'),
            name=asset
        ))

    fig.update_layout(
        title="Efficient frontier and individual assets",
        xaxis_title="Annual volatility",
        yaxis_title="Annual return"
    )

    st.plotly_chart(fig)

def plot_sector_allocation(allocation):
    sectors = {
        "ENGI.PA": "Énergie", "BNP.PA": "Finance", "ACA.PA": "Industrie",
        "GLE.PA": "Énergie", "TTE.PA": "Énergie", "MC.PA": "Luxe",
        "OR.PA": "Mines", "AIR.PA": "Industrie", "RNO.PA": "Automobile", "VK.PA": "Technologie"
    }
    sector_allocation = allocation.assign(Secteur=lambda x: x.index.map(sectors)).groupby("Secteur").sum()
    st.bar_chart(sector_allocation)

def plot_correlation_matrix(returns):
    corr_matrix = returns.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu"
    ))
    st.plotly_chart(fig)
