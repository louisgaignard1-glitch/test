import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from data_utils import get_data
from optimization import optimize_portfolio
from visualization import plot_efficient_frontier, plot_sector_allocation, plot_correlation_matrix
from metrics import display_metrics
from backtest import backtest_portfolio
from monte_carlo import monte_carlo_simulation

# Page config
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# Title
st.title("📈 Louis GAIGNARD Portfolio Optimization with Python (Markowitz, Monte Carlo...)")

# Asset selection
assets = st.multiselect(
    "Select your assets",
    options=["ENGI.PA", "BNP.PA", "ACA.PA", "GLE.PA", "TTE.PA", "MC.PA", "OR.PA", "AIR.PA", "RNO.PA", "VK.PA"],
    default=["ENGI.PA", "BNP.PA", "TTE.PA", "MC.PA"]
)

# Validation
if len(assets) < 2:
    st.warning("Please select at least 2 assets to run the optimization.")
    st.stop()

# Data
data = get_data(assets, "2022-01-01")
if data.empty:
    st.error("No data available for the selected assets and dates.")
    st.stop()

# Returns
returns = data.pct_change().dropna()
mu = returns.mean() * 252
Sigma = returns.cov() * 252

# Optimization
result_min_var, allocation = optimize_portfolio(mu, Sigma, assets)

# Manual allocation
st.header("🎛️ Manual Allocation")

manual_weights = {}
cols = st.columns(len(assets))

for i, asset in enumerate(assets):
    manual_weights[asset] = cols[i].slider(
        asset,
        0.0,
        1.0,
        float(allocation.loc[asset, "Poids"]),
        key=f"slider_{asset}"
    )

manual_weights = pd.Series(manual_weights)

st.write(f"Weight sum: {manual_weights.sum():.2f}")

if st.button("Normalize weights"):
    manual_weights = manual_weights / manual_weights.sum()

manual_allocation = pd.DataFrame({"Poids": manual_weights})

# Toggle manual allocation
use_manual = st.toggle("Use manual allocation", True)

weights_used = manual_allocation if use_manual else allocation

# Allocation comparison
fig_alloc = go.Figure()

fig_alloc.add_trace(go.Bar(
    x=allocation.index,
    y=allocation["Poids"],
    name="Optimized"
))

fig_alloc.add_trace(go.Bar(
    x=manual_allocation.index,
    y=manual_allocation["Poids"],
    name="Manual"
))

fig_alloc.update_layout(
    barmode="group",
    title="Allocation comparison",
    yaxis_title="Weight"
)

st.plotly_chart(fig_alloc, use_container_width=True)

# Metrics
display_metrics(result_min_var, mu, Sigma, returns)

# Efficient frontier
plot_efficient_frontier(result_min_var, mu, Sigma, assets)

# Sector allocation
st.header("📊 Sector Allocation")
plot_sector_allocation(weights_used)

# Correlation matrix
st.header("🔗 Asset Correlation Matrix")
plot_correlation_matrix(returns)

# Backtest
st.header("📈 Historical Cumulative Performance")
portfolio_performance = backtest_portfolio(data, weights_used)
st.line_chart(portfolio_performance)

# Monte Carlo
st.header("🔮 Monte Carlo Simulation (Fan Chart, VaR & CVaR)")
n_sim = st.sidebar.slider("Monte Carlo simulations", 100, 800, 300)
percentiles, var_95, cvar_95 = monte_carlo_simulation(mu, Sigma, weights_used)

col1, col2 = st.columns(2)
col1.metric("VaR 95%", f"{var_95:.2%}")
col2.metric("CVaR 95%", f"{cvar_95:.2%}")

fig = go.Figure()

fig.add_trace(go.Scatter(x=percentiles["p95"].index, y=percentiles["p95"],
                         line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=percentiles["p5"].index, y=percentiles["p5"],
                         fill='tonexty', name='5-95%', line=dict(width=0)))

fig.add_trace(go.Scatter(x=percentiles["p75"].index, y=percentiles["p75"],
                         line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=percentiles["p25"].index, y=percentiles["p25"],
                         fill='tonexty', name='25-75%', line=dict(width=0)))

fig.add_trace(go.Scatter(x=percentiles["p50"].index, y=percentiles["p50"],
                         name='Median', line=dict(width=3)))

fig.update_layout(
    title="Probabilistic Fan Chart",
    hovermode="x unified",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
