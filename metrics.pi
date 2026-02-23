
import streamlit as st

def display_metrics(result_min_var, mu, Sigma, returns):
    if result_min_var.success:
        from optimization import calculate_sharpe_ratio, calculate_max_drawdown

        weights = result_min_var.x
        portfolio_returns = returns.dot(weights)

        sharpe_ratio = calculate_sharpe_ratio(weights, mu, Sigma)
        max_drawdown = calculate_max_drawdown(portfolio_returns)

        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
