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
