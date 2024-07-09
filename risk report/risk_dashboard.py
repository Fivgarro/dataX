import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_var(stock_returns, confidence_level=0.95):
    mean = np.mean(stock_returns)
    std_dev = np.std(stock_returns)
    var = norm.ppf(1 - confidence_level, mean, std_dev)
    return var

def calculate_es(stock_returns, confidence_level=0.95):
    var = calculate_var(stock_returns, confidence_level)
    es = stock_returns[stock_returns <= var].mean()
    return es

def stress_test(stock_data, shock_factor=0.2):
    stressed_data = stock_data * (1 - shock_factor)
    return stressed_data

def backtest_var(stock_returns, var):
    breaches = stock_returns[stock_returns < var]
    breach_percentage = len(breaches) / len(stock_returns)
    return breach_percentage

st.title("Market Risk Dashboard")
st.sidebar.header("User Inputs")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

tickers = ["AAPL", "MSFT", "GOOGL"]
selected_stocks = st.sidebar.multiselect("Select Stocks", tickers, default=tickers)

if st.sidebar.button("Load Data"):
    stock_data = get_stock_data(selected_stocks, start_date, end_date)
    st.write(stock_data)

    returns = stock_data.pct_change().dropna()
    st.write("Stock Returns:", returns)

    var_values = returns.apply(calculate_var)
    st.write("VaR Values (95% Confidence Level):", var_values)

    es_values = returns.apply(calculate_es)
    st.write("Expected Shortfall (95% Confidence Level):", es_values)

    shocked_stock_data = stress_test(stock_data)
    st.write("Stressed Data (20% Shock):", shocked_stock_data)

    # backtest_results = returns.apply(backtest_var, args=(var_values,))
    # st.write("Backtesting Results:", backtest_results)

    st.line_chart(stock_data, width=0, height=0)
    st.line_chart(returns, width=0, height=0)
    
    #breaches = returns[returns < var_values]
    #st.line_chart(breaches, width=0, height=0)
