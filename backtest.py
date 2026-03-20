def backtest_portfolio(data, allocation):
    returns = data.pct_change().dropna()
    weights = allocation['Poids'].reindex(returns.columns).fillna(0)
    portfolio_returns = returns.dot(weights)
    cumulative = (1 + portfolio_returns).cumprod()


    annualized_return = portfolio_returns.mean() * 252
    annualized_vol = portfolio_returns.std() * np.sqrt(252)
    
    return cumulative, annualized_return, annualized_vol
    
benchmark = yf.download("^FCHI", start=start_date, auto_adjust=True)["Close"]
benchmark_returns = (1 + benchmark.pct_change().dropna()).cumprod()
