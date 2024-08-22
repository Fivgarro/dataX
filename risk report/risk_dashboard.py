import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

# Function to determine the financial quarter based on the date
def get_financial_quarter(date):
    month = date.month
    if month in [7, 8, 9]:
        return "Q1: 1 July to 30 September"
    elif month in [10, 11, 12]:
        return "Q2: 1 October to 31 December"
    elif month in [1, 2, 3]:
        return "Q3: 1 January to 31 March"
    elif month in [4, 5, 6]:
        return "Q4: 1 April to 30 June"

# User Inputs for Reporting
st.title("ARS 116 - Internal Model Approach")
st.write("")
st.sidebar.header("Reporting Inputs")

# Date input for reporting period
reporting_date = st.sidebar.date_input("Select Reporting Date", datetime.today())
financial_quarter = get_financial_quarter(reporting_date)

# reporting_consolidation = st.sidebar.selectbox("Reporting Consolidation", ["Level 1", "Level 2"])

# Function to calculate daily VaR and stressed VaR
def calculate_var(returns, confidence_level=0.99):
    mean = returns.mean()
    std_dev = returns.std()
    var = norm.ppf(1 - confidence_level, mean, std_dev)
    return var

def calculate_stressed_var(returns, stress_factor=1.5, confidence_level=0.99):
    stressed_returns = returns * stress_factor
    stressed_var = calculate_var(stressed_returns, confidence_level)
    return stressed_var

def backtest_var(returns, var_series):
    exceptions = returns[returns < -var_series]
    return len(exceptions)

# Example stock data import and processing
tickers = ["AAPL", "MSFT", "GOOGL"]
selected_stocks = st.sidebar.multiselect("Select Stocks", tickers, default=tickers)
data = yf.download(selected_stocks, start="2022-07-01", end="2023-06-30")['Adj Close']

# Calculate returns
returns = data.pct_change().dropna()

# Column 1: End of quarter VaR
end_of_quarter_var = abs(calculate_var(returns.iloc[-1])* 1e8)

# Column 2: Average VaR over past 60 trading days
average_var_60_days = returns.rolling(window=60).apply(calculate_var).iloc[-1].mean()

# Column 3: End of quarter stressed VaR
end_of_quarter_stressed_var = calculate_stressed_var(returns.iloc[-1])

# Column 4: Average stressed VaR over past 60 trading days
average_stressed_var_60_days = returns.rolling(window=60).apply(calculate_stressed_var).iloc[-1].mean()

# Columns 5 & 6: Back-testing exceptions (last 250 trading days)
backtest_exceptions_actual = backtest_var(returns.iloc[-250:], returns.apply(calculate_var))
backtest_exceptions_hypothetical = backtest_var(returns.iloc[-250:], returns.apply(lambda x: calculate_var(x) * 1.1)) # Hypothetical adjustment

# Columns 7 & 8: Scaling factors (manually set, or provided by APRA)
scaling_factor_var = 3.0  # Example value
scaling_factor_stressed_var = 3.0  # Example value

# Columns 9 & 10: Scaled average VaR and stressed VaR
scaled_avg_var = average_var_60_days * scaling_factor_var
scaled_avg_stressed_var = average_stressed_var_60_days * scaling_factor_stressed_var

# Headline information and measurements
info = {
    "Financial Quarter": [financial_quarter],
    "Scale Factor:": ["Millions to two decimal places"]
}

# Prepare data for table
report_data = {
    "End of quarter VaR": [end_of_quarter_var],
    "Average VaR over past 60 trading days": [average_var_60_days],
    "End of quarter stressed VaR": [end_of_quarter_stressed_var],
    "Average stressed VaR over past 60 trading days": [average_stressed_var_60_days],
    "Back-testing exceptions (actual)": [backtest_exceptions_actual],
    "Back-testing exceptions (hypothetical)": [backtest_exceptions_hypothetical],
    "Scaling factor (VaR)": [scaling_factor_var],
    "Scaling factor (stressed VaR)": [scaling_factor_stressed_var],
    "Scaled average VaR": [scaled_avg_var],
    "Scaled average stressed VaR": [scaled_avg_stressed_var],
}

#for i in range(6):
#   if i not in report_data[
#  "Back-testing exceptions (actual)","Back-testing exceptions (hypothetical)",
# "Scaling factor (VaR)","Scaling factor (stressed VaR)"
#]:
#        report_data = abs(round(i))

# Convert to DataFrame for display
df_info = pd.DataFrame(info)
df_report = pd.DataFrame(report_data)

# Display the table in Streamlit
st.table(df_info)
st.table(df_report)
