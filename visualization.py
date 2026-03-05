import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy.optimize import minimize


def plot_efficient_frontier(result_min_var, mu, Sigma, assets):

    target_returns = np.linspace(mu.min(), mu.max(), 20)

    frontier_vols = []

    for target_return in target_returns:

        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, tr=target_return: np.dot(w, mu) - tr}
        )

        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(Sigma, w))),
            np.ones(len(assets)) / len(assets),
            method='SLSQP',
            bounds=tuple((0, 1) for _ in assets),
            constraints=constraints
        )

        if result.success:

            frontier_vols.append(
                np.sqrt(np.dot(result.x.T, np.dot(Sigma, result.x)))
            )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier_vols,
        y=target_returns,
        mode='lines+markers'
    ))

    st.plotly_chart(fig)



def plot_sector_allocation(allocation):

    sectors = {
        "ENGI.PA": "Énergie",
        "BNP.PA": "Finance",
        "ACA.PA": "Finance",
        "GLE.PA": "Finance",
        "TTE.PA": "Énergie",
        "MC.PA": "Luxe",
        "OR.PA": "Consommation",
        "AIR.PA": "Aéronautique",
        "RNO.PA": "Automobile",
        "VK.PA": "Industrie",

        "KER.PA": "Luxe",
        "RMS.PA": "Luxe",

        "SAF.PA": "Aéronautique",
        "HO.PA": "Défense",

        "SU.PA": "Industrie",
        "CAP.PA": "Technologie",
        "STMPA.PA": "Technologie",

        "EDF.PA": "Énergie",
        "VIE.PA": "Environnement",
        "EN.PA": "Construction",

        "SAN.PA": "Santé",

        "SGO.PA": "Matériaux",

        "ORA.PA": "Télécom",

        "CA.PA": "Distribution",
        "RI.PA": "Consommation",

        "DG.PA": "Construction",
        "AI.PA": "Industrie"
    }

    sector_allocation = allocation.copy()

    sector_allocation["Secteur"] = sector_allocation.index.map(sectors)

    # si un ticker n'est pas reconnu
    sector_allocation["Secteur"] = sector_allocation["Secteur"].fillna("Autre")

    sector_allocation = sector_allocation.groupby("Secteur")["Poids"].sum()

    st.bar_chart(sector_allocation)

    sector_allocation = allocation.assign(
        Secteur=lambda x: x.index.map(sectors)
    ).groupby("Secteur").sum()

    st.bar_chart(sector_allocation)


def plot_correlation_matrix(returns):

    corr_matrix = returns.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns
        )
    )

    st.plotly_chart(fig)
