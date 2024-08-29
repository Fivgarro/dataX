# pip install transformers==4.30.2 torch==2.3.0
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

# Setup layout
st.set_page_config(page_title="ARS 116 - Internal Model Approach", layout="wide", initial_sidebar_state="collapsed")
col1, col2 = st.columns((2, 1), gap = "small")

# User Inputs for Reporting
col1.title("ARS 116.0 - Internal Model Approach")

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
end_of_quarter_var = calculate_var(returns.iloc[-1])

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
scaling_factor_var = 3  # Example value
scaling_factor_stressed_var = 3  # Example value

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
    "": "general market risk",
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

formatted_data = ["End of quarter VaR","Average VaR over past 60 trading days",
            "End of quarter stressed VaR","Average stressed VaR over past 60 trading days",
            "Scaled average VaR","Scaled average stressed VaR"
]

# Update the values in report_data directly
for k in formatted_data:
    report_data[k] = [round(abs(report_data[k][0] * 1e8), 2)]

# Convert to DataFrame for display
df_info = pd.DataFrame(info)
df_report = pd.DataFrame(report_data)

# Convert report_data to a string for the model to read
report_text = "\n".join([f"{key}: {value[0]}" for key, value in report_data.items()])


# FinGPTWrapper class definition
class FinGPTWrapper:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, prompt):
        template = "Instruction: Generate a financial risk report summary based on the input. Input: {} Answer:"
        full_prompt = template.format(prompt)
        inputs = self.tokenizer(full_prompt, return_tensors='pt', padding=True)
        outputs = self.model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            max_new_tokens=50,  # Limit to 50 new tokens
            temperature=0.3,    # Control randomness
            top_p=0.9,          # Use nucleus sampling
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # because of the continual repetition of the prompt template im extracting the part after 'Answer:'
        response = response.split("Answer:")[1].strip()
        return response

# Initialize the FinGPTWrapper
fin_gpt = FinGPTWrapper()

# Generate the summary
summary = fin_gpt.generate_response(report_text)


# Display the table in Streamlit
col1.dataframe(df_info, hide_index = True)
col1.write("")
col1.write("")
col1.dataframe(df_report, hide_index = True)
col2.header("Summary")
col2.write(summary)
