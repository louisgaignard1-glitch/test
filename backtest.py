def backtest_portfolio(data, allocation):
    returns = data.pct_change().fillna(0)
    weights = allocation['Poids'].reindex(returns.columns).fillna(0).values
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns
