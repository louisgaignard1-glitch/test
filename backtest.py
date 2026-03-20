def backtest_portfolio(data, allocation):

    returns = data.pct_change().dropna()

    weights = allocation['Poids'].reindex(returns.columns).fillna(0)

    portfolio_returns = returns.dot(weights)

    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    benchmark = yf.download("^FCHI", start=start_date, auto_adjust=True)["Close"]
    benchmark_returns = (1 + benchmark.pct_change().dropna()).cumprod()

    return cumulative_returns
