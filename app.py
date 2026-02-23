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

import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data
def get_data(tickers, start_date):
    try:
        data = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                data = data["Close"]
            elif "Adj Close" in data.columns.levels[0]:
                data = data["Adj Close"]

        return data.dropna(how="all")
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return pd.DataFrame()
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_portfolio(mu, Sigma, assets):
    n = len(assets)

    def port_return(weights):
        return np.dot(weights, mu)

    def port_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))

    initial_weights = np.ones(n) / n
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))

    result_min_var = minimize(
        port_vol,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    allocation = pd.DataFrame({'Poids': result_min_var.x}, index=assets)


    return result_min_var, allocation

def calculate_sharpe_ratio(weights, mu, Sigma, risk_free_rate=0.02):
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
    return float((port_return - risk_free_rate) / port_vol)  # Assurez-vous de retourner un float

def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return float(drawdown.min())  # Assurez-vous de retourner un float


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



import streamlit as st

def display_metrics(result_min_var, mu, Sigma, returns):
    if result_min_var.success:
        from utils.optimization import calculate_sharpe_ratio, calculate_max_drawdown

        weights = result_min_var.x
        portfolio_returns = returns.dot(weights)

        sharpe_ratio = calculate_sharpe_ratio(weights, mu, Sigma)
        max_drawdown = calculate_max_drawdown(portfolio_returns)

        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")



import pandas as pd

def backtest_portfolio(data, allocation):
    returns = data.pct_change().dropna()

    # alignement des poids
    weights = allocation['Poids'].reindex(returns.columns).fillna(0)

    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    return cumulative_returns



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
