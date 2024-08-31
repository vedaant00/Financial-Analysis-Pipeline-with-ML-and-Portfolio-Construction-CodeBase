# # -*- coding: utf-8 -*-
# !python --version

# !pip install wrds
# !pip install --upgrade numpy scipy pyarrow pandas datasets transformers
# !pip install accelerate -U

# """## Check GPU availability"""

# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#   print('Not connected to a GPU')
# else:
#   print(gpu_info)

"""## Importing all necessary libraries and modules"""

import os
import ast
import torch
import shutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import Counter, defaultdict
from datetime import timedelta
from wordcloud import WordCloud

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback

import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt')

warnings.filterwarnings('ignore')

"""## Loading Datasets and Preparing for Model Application

This section outlines the initial steps for loading the in-sample and out-sample datasets, followed by the preparation required for applying various models.

### Key Steps

1. **Loading the Cleaned Datasets**:
   - The cleaned datasets, `cleaned_in_sample_data.csv` and `cleaned_out_sample_data.csv`, were loaded into `in_sample_data` and `out_sample_data` respectively.

2. **Model Name Mapping**:
   - A dictionary (`model_name_mapping`) was created to map descriptive model names to their respective pretrained model identifiers.
   - This mapping includes popular models such as BERT, RoBERTa, DistilBERT, DistilRoBERTa, and FinBERT.

3. **Ensuring 'Start_Date' is in Datetime Format**:
   - The `Start_Date` column in both datasets was converted to datetime format to ensure that the date-based operations, especially those crucial for time-series analysis and model training, are performed accurately.
"""

# Load in-sample and out-sample datasets
in_sample_data = pd.read_csv('/content/cleaned_in_sample_data.csv')
out_sample_data = pd.read_csv('/content/cleaned_out_sample_data.csv')

# Define model name mapping
model_name_mapping = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased',
    'distilroberta': 'distilroberta-base',
    'finbert': 'yiyanghkust/finbert-tone'
}

# Ensure 'Start_Date' is in datetime format
in_sample_data['Start_Date'] = pd.to_datetime(in_sample_data['Start_Date'])
out_sample_data['Start_Date'] = pd.to_datetime(out_sample_data['Start_Date'])

"""## Portfolio Construction and Analysis: Market Capitalization and Portfolio Returns

This section outlines the process of constructing and analyzing a stock portfolio based on market capitalization. The workflow includes data preprocessing, market cap classification, and portfolio return calculations with and without transaction costs.

### Key Steps in Portfolio Construction and Analysis

1. **Loading and Preprocessing Market Capitalization Data**:
   - **Data Loading**:
     - The market capitalization data is loaded from a CSV file, with specific data types assigned to relevant columns to handle large datasets and potential errors.
     - **`PRC`** and **`SHROUT`** columns are converted to numeric values, with errors forced to **NaN** to handle non-numeric data.

   - **Data Cleaning**:
     - Rows with **NaN** values in **`PRC`** or **`SHROUT`** columns are dropped.
     - The data is filtered to include only NYSE stocks, identified by **`EXCHCD == 1`**.

   - **Date Handling**:
     - The **`date`** column is converted to datetime format, and timezone information is removed if necessary to ensure consistency.
     - Rows where the date conversion failed are dropped.

   - **Market Cap Calculation**:
     - Market capitalization is calculated as **`PRC`** multiplied by **`SHROUT`**.

   - **Stock Classification**:
     - Stocks are classified as **Large** or **Small** based on whether their market capitalization is above or below the median for that date.
     - The **`TICKER`** column is renamed to **`Ticker`** for consistency.

2. **Loading and Preprocessing Portfolio Returns Data**:
   - The returns data is loaded from a CSV file and preprocessed to calculate daily returns based on the adjusted price.
   - The data is filtered to include only the relevant time period (2016-2024) and rows with valid daily returns.

3. **Merging Market Cap Data with Returns Data**:
   - The market capitalization data is merged with the returns data based on **`Start_Date`** and **`Ticker`**.
   - Transaction costs are applied to the returns based on the stock classification (Large or Small).
   - Adjusted returns are calculated by subtracting the transaction costs from the daily returns.

4. **Aggregating Data by Week**:
   - The daily returns are aggregated by week, and the returns are calculated for each week for both the original and adjusted returns.
   - This weekly aggregation is crucial for analyzing portfolio performance over time.

5. **Pivoting Data for Portfolio Analysis**:
   - The data is pivoted to create a matrix where each row represents a week and each column represents a stock ticker. The values are the weekly returns.
   - The same pivoting is done for both the original and adjusted returns.

   - **Market Returns Calculation**:
     - The market return is assumed to be the average of all returns for each period, providing a benchmark for portfolio performance.

6. **Saving Results**:
   - The weekly returns without transaction costs are saved to a CSV file.
   - The weekly returns with transaction costs are also saved to a separate CSV file.
"""

# Step 1: Load the data and specify dtype for relevant columns
market_cap_data = pd.read_csv('/content/crsp_2016-2023.csv',
                              dtype={'PRC': 'str', 'SHROUT': 'str'},
                              on_bad_lines='skip')

# Convert PRC and SHROUT to numeric, forcing errors to NaN
market_cap_data['PRC'] = pd.to_numeric(market_cap_data['PRC'], errors='coerce')
market_cap_data['SHROUT'] = pd.to_numeric(market_cap_data['SHROUT'], errors='coerce')

# Drop rows where PRC or SHROUT couldn't be converted to numbers
market_cap_data.dropna(subset=['PRC', 'SHROUT'], inplace=True)

# Step 2: Filter for NYSE stocks (EXCHCD == 1)
market_cap_data = market_cap_data[market_cap_data['EXCHCD'] == 1]

# Step 3: Handle the date column and ensure consistency
market_cap_data['date'] = pd.to_datetime(market_cap_data['date'], errors='coerce')

# Remove timezone information to make all dates naive (if necessary)
market_cap_data['date'] = market_cap_data['date'].dt.tz_localize(None)

# Drop rows where date conversion failed
market_cap_data.dropna(subset=['date'], inplace=True)

# Step 4: Calculate Market Cap
market_cap_data['Market_Cap'] = market_cap_data['PRC'] * market_cap_data['SHROUT']

# Step 5: Classify Stocks Based on Market Cap
market_cap_data['Classification'] = market_cap_data.groupby('date')['Market_Cap'].transform(
    lambda x: np.where(x >= x.median(), 'Large', 'Small')
)

# Rename TICKER to Ticker
market_cap_data.rename(columns={'TICKER': 'Ticker'}, inplace=True)

# Step 6: Load Portfolio Returns Data
returns_data = pd.read_csv('/content/cleaned_out_sample_data.csv')

# Data preprocessing
returns_data['Start_Date'] = pd.to_datetime(returns_data['Start_Date'])
returns_data['Daily_Return'] = returns_data.groupby('Ticker')['Adjusted_Price'].pct_change()
filtered_data = returns_data[(returns_data['Start_Date'] >= '2016-01-01') & (returns_data['Start_Date'] <= '2024-12-31')]
filtered_data = filtered_data.dropna(subset=['Daily_Return'])

# Step 7: Merge Market Cap Data with Returns Data
merged_data = filtered_data.merge(market_cap_data[['date', 'Ticker', 'Classification']],
                                  left_on=['Start_Date', 'Ticker'],
                                  right_on=['date', 'Ticker'],
                                  how='left')

# Step 8: Apply Transaction Costs
merged_data['Transaction_Cost'] = np.where(
    merged_data['Classification'] == 'Large', 11.21 / 10000, 21.27 / 10000
)
merged_data['Adjusted_Return'] = merged_data['Daily_Return'] - merged_data['Transaction_Cost']

# Step 9: Aggregate Data by Week
merged_data['Week'] = merged_data['Start_Date'].dt.to_period('W')
aggregated_data = merged_data.groupby(['Week', 'Ticker']).agg({'Daily_Return': lambda x: (1 + x).prod() - 1}).reset_index()
adjusted_aggregated_data = merged_data.groupby(['Week', 'Ticker']).agg({'Adjusted_Return': lambda x: (1 + x).prod() - 1}).reset_index()

# Step 10: Pivot Data for Portfolio Analysis
returns_data_pivot = aggregated_data.pivot(index='Week', columns='Ticker', values='Daily_Return')
adjusted_returns_data_pivot = adjusted_aggregated_data.pivot(index='Week', columns='Ticker', values='Adjusted_Return')

returns_data_pivot.index = returns_data_pivot.index.to_timestamp()
adjusted_returns_data_pivot.index = adjusted_returns_data_pivot.index.to_timestamp()

# Assuming the market return is the average of all returns for each period
market_returns = returns_data_pivot.mean(axis=1)
adjusted_market_returns = adjusted_returns_data_pivot.mean(axis=1)

# Save the returns without transaction costs to a CSV file
returns_data_pivot.to_csv('/content/returns_without_transaction_costs.csv')

# Save the returns with transaction costs to a CSV file
adjusted_returns_data_pivot.to_csv('/content/returns_with_transaction_costs.csv')

"""## Common Functions for Portfolio Construction and Analysis

This section details the common functions used for portfolio construction and analysis in both rolling-window base models and fine-tuned models. These functions are essential for calculating portfolio returns, assessing performance, and evaluating the effectiveness of different investment strategies.

### Key Functions for Portfolio Analysis

1. **Calculate Returns for a Portfolio**:
    - **`calculate_returns(data, weights)`**:
        - This function calculates the weighted returns of a portfolio based on the given data and weights.
        - **`data`**: A DataFrame where each column represents a stock's returns.
        - **`weights`**: An array of weights assigned to each stock.
        - The function multiplies the stock returns by their respective weights and sums them to get the overall portfolio returns.

2. **Portfolio Returns Calculation Based on Predictions**:
    - **`calculate_portfolio_returns(data, predictions, N=10)`**:
        - This function calculates the returns for portfolios constructed based on predictions.
        - **`N`**: The number of top and bottom stocks to include in the long (L) and short (S) portfolios.
        - **Equally Weighted (EW) Portfolios**:
            - **`ew_l`**: Returns for the top **`N`** stocks.
            - **`ew_s`**: Returns for the bottom **`N`** stocks.
            - **`ew_ls`**: Long-short portfolio returns (top **`N`** minus bottom **`N`**).
        - **Value Weighted (VW) Portfolios**:
            - **`vw_l`**: Returns for the top **`N`** stocks, weighted by their prediction scores.
            - **`vw_s`**: Returns for the bottom **`N`** stocks, weighted by their prediction scores.
            - **`vw_ls`**: Long-short portfolio returns (top **`N`** minus bottom **`N`**).

3. **Sharpe Ratio Calculation**:
    - **`calculate_sharpe_ratio(returns, risk_free_rate=0)`**:
        - This function calculates the Sharpe ratio, which is a measure of risk-adjusted return.
        - **`returns`**: The portfolio returns.
        - **`risk_free_rate`**: The risk-free rate, defaulting to 0.
        - The function calculates the excess returns (returns minus risk-free rate) and divides the mean excess return by its standard deviation to obtain the Sharpe ratio.

4. **Process Model for Portfolio Construction**:
    - **`process_model(model, predicted_directions, returns_data, market_returns)`**:
        - This function processes a specific model to calculate the portfolio returns based on the model's predictions.
        - **`model`**: The model being evaluated.
        - **`predicted_directions`**: DataFrame containing the predicted directions for each stock.
        - **`returns_data`**: DataFrame containing the actual returns data for the stocks.
        - **`market_returns`**: Benchmark market returns for comparison.
        - The function filters the relevant data for the given model, calculates predicted returns, and constructs portfolios (both equally weighted and value weighted).
        - It returns the portfolio returns for both equally weighted and value weighted strategies, along with the overall portfolio returns and market returns.
"""

# Step 11: Perform Portfolio Analysis
def calculate_returns(data, weights):
    weighted_returns = data.mul(weights, axis=1)
    portfolio_returns = weighted_returns.sum(axis=1)
    return portfolio_returns

def calculate_portfolio_returns(data, predictions, N=10):
    top_n_indices = predictions.argsort()[-N:][::-1]
    bottom_n_indices = predictions.argsort()[:N]

    ew_weights = np.array([1/N] * N)
    vw_weights_top = predictions[top_n_indices] / predictions[top_n_indices].sum()
    vw_weights_bottom = predictions[bottom_n_indices] / predictions[bottom_n_indices].sum()

    ew_l = calculate_returns(data.iloc[:, top_n_indices], ew_weights)
    ew_s = calculate_returns(data.iloc[:, bottom_n_indices], ew_weights)
    ew_ls = ew_l - ew_s

    vw_l = calculate_returns(data.iloc[:, top_n_indices], vw_weights_top)
    vw_s = calculate_returns(data.iloc[:, bottom_n_indices], vw_weights_bottom)
    vw_ls = vw_l - vw_s

    returns = {
        'EW L': ew_l,
        'EW S': ew_s,
        'EW LS': ew_ls,
        'VW L': vw_l,
        'VW S': vw_s,
        'VW LS': vw_ls
    }
    return returns

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

def process_model(model, predicted_directions, returns_data, market_returns):
    model_data = predicted_directions[predicted_directions['Model'] == model]
    model_tickers = model_data['Ticker'].unique()
    model_returns_data = returns_data.loc[:, returns_data.columns.isin(model_tickers)]

    if model_returns_data.empty:
        print(f"No data available for model: {model}")
        return None, None, None

    predicted_returns = model_data.groupby('Ticker')['Predicted_Direction'].mean()
    portfolio_returns = calculate_portfolio_returns(model_returns_data, predicted_returns)

    ew_l_returns = portfolio_returns['EW L']
    ew_s_returns = portfolio_returns['EW S']
    ew_ls_returns = portfolio_returns['EW LS']

    vw_l_returns = portfolio_returns['VW L']
    vw_s_returns = portfolio_returns['VW S']
    vw_ls_returns = vw_l_returns - vw_s_returns

    return ew_l_returns, ew_s_returns, ew_ls_returns, vw_l_returns, vw_s_returns, vw_ls_returns, portfolio_returns, market_returns

"""## Processing Un-Tuned Model Predictions for Portfolio Analysis

In this section, we detail the steps for processing and analyzing the un-tuned model predictions using portfolio construction techniques. This involves calculating portfolio returns, Sharpe ratios, and visualizing cumulative returns over time.

### Key Steps in the Process

1. **Loading and Iterating Over Un-Tuned Model Predictions**:
    - **`un_tuned_predicted_directions`**: The DataFrame containing the predicted directions for each stock based on un-tuned models is loaded from a CSV file.
    - **`un_tuned_models`**: Unique model identifiers are extracted to iterate over each model’s predictions.
    - For each model, the following steps are performed to calculate portfolio returns and metrics.

2. **Processing Each Model**:
    - **`process_model()`**: This function is used to calculate portfolio returns for both equally weighted (EW) and value weighted (VW) portfolios:
        - **EW and VW Long**: Returns for the top N stocks.
        - **EW and VW Short**: Returns for the bottom N stocks.
        - **EW and VW Long-Short (LS)**: Returns for a portfolio that goes long on the top N stocks and short on the bottom N stocks.
    - **Transaction Costs**: Portfolio returns with and without transaction costs are considered, though in this case, the adjusted returns are commented out.

3. **Calculating and Storing Sharpe Ratios**:
    - **`calculate_sharpe_ratio()`**: Sharpe ratios are calculated for each portfolio configuration (EW L, EW S, EW LS, VW L, VW S, VW LS) as well as for the market benchmark.
    - The results are stored in a list of dictionaries, where each dictionary contains the Sharpe ratios for a specific model.

4. **Visualizing Cumulative Returns**:
    - **Cumulative Returns Plot**: For each model, cumulative returns are plotted over time to visualize the performance of the portfolios.
        - The cumulative log returns for each portfolio and the market benchmark are plotted, with different line styles distinguishing between unadjusted and adjusted returns.

5. **Saving the Sharpe Ratios**:
    - **Conversion to DataFrame**: The list of Sharpe ratios is converted into a DataFrame for easier analysis and saving.
    - **Saving to CSV**: The resulting DataFrame containing Sharpe ratios for all models is saved to a CSV file for further analysis.
"""

# Step 12: Process each model's un-tuned predictions
un_tuned_predicted_directions = pd.read_csv('/content/predicted_rolling_window_test_directions.csv')
un_tuned_models = un_tuned_predicted_directions['Model'].unique()
un_tuned_all_sharpe_ratios = []

for model in un_tuned_models:
    ew_l, ew_s, ew_ls, vw_l, vw_s, vw_ls, portfolio_returns, market_returns = process_model(
        model, un_tuned_predicted_directions, returns_data_pivot, market_returns)
    adj_ew_l, adj_ew_s, adj_ew_ls, adj_vw_l, adj_vw_s, adj_vw_ls, adj_portfolio_returns, adj_market_returns = process_model(
        model, un_tuned_predicted_directions, adjusted_returns_data_pivot, adjusted_market_returns)

    if ew_l is None or vw_l is None:
        continue

    model_sharpe_ratios = {
        'Model': model,
        'EW L': calculate_sharpe_ratio(ew_l),
        'EW S': calculate_sharpe_ratio(ew_s),
        'EW LS': calculate_sharpe_ratio(ew_ls),
        'VW L': calculate_sharpe_ratio(vw_l),
        'VW S': calculate_sharpe_ratio(vw_s),
        'VW LS': calculate_sharpe_ratio(vw_ls),
        'Market': calculate_sharpe_ratio(market_returns),
        # 'Adj EW L': calculate_sharpe_ratio(adj_ew_l),
        # 'Adj EW S': calculate_sharpe_ratio(adj_ew_s),
        # 'Adj EW LS': calculate_sharpe_ratio(adj_ew_ls),
        # 'Adj VW L': calculate_sharpe_ratio(adj_vw_l),
        # 'Adj VW S': calculate_sharpe_ratio(adj_vw_s),
        # 'Adj VW LS': calculate_sharpe_ratio(adj_vw_ls),
        # 'Adj Market': calculate_sharpe_ratio(adj_market_returns)
    }

    un_tuned_all_sharpe_ratios.append(model_sharpe_ratios)

    # Plotting Cumulative Returns
    plt.figure(figsize=(12, 8))
    for label, returns in portfolio_returns.items():
        plt.plot((1 + returns).cumprod() - 1, label=label)
    # for label, returns in adj_portfolio_returns.items():
    #     plt.plot((1 + returns).cumprod() - 1, label=label, linestyle='--')
    plt.plot((1 + market_returns).cumprod() - 1, label='Market', linestyle=':')
    # plt.plot((1 + adj_market_returns).cumprod() - 1, label='Adj Market', linestyle='-.')
    plt.title(f'Cumulative Weekly Portfolio Returns Over Time for Un-Tuned {model}')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Log Returns', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 13: Convert Sharpe ratios to DataFrame and save
un_tuned_sharpe_ratios_df = pd.DataFrame(un_tuned_all_sharpe_ratios)
print(un_tuned_sharpe_ratios_df)

# Save the results to a CSV file
un_tuned_sharpe_ratios_df.to_csv('/content/un_tuned_sharpe_ratios.csv', index=False)

"""## Processing Fine-Tuned Model Predictions for Portfolio Analysis

This section outlines the process of analyzing fine-tuned model predictions by constructing portfolios, calculating Sharpe ratios, and visualizing cumulative returns over time.

### Key Steps in the Process

1. **Loading and Iterating Over Fine-Tuned Model Predictions**:
    - **`fine_tuned_predicted_directions`**: The DataFrame containing the predicted directions for each stock based on fine-tuned models is loaded from a CSV file.
    - **`fine_tuned_models`**: Unique model identifiers are extracted to iterate over each model’s predictions.
    - For each model, the following steps are performed to calculate portfolio returns and metrics.

2. **Processing Each Fine-Tuned Model**:
    - **`process_model()`**: This function is used to calculate portfolio returns for both equally weighted (EW) and value weighted (VW) portfolios:
        - **EW and VW Long**: Returns for the top N stocks.
        - **EW and VW Short**: Returns for the bottom N stocks.
        - **EW and VW Long-Short (LS)**: Returns for a portfolio that goes long on the top N stocks and short on the bottom N stocks.
    - **Transaction Costs**: Portfolio returns with and without transaction costs are considered, though in this case, the adjusted returns are commented out.

3. **Calculating and Storing Sharpe Ratios**:
    - **`calculate_sharpe_ratio()`**: Sharpe ratios are calculated for each portfolio configuration (EW L, EW S, EW LS, VW L, VW S, VW LS) as well as for the market benchmark.
    - The results are stored in a list of dictionaries, where each dictionary contains the Sharpe ratios for a specific model.

4. **Visualizing Cumulative Returns**:
    - **Cumulative Returns Plot**: For each model, cumulative returns are plotted over time to visualize the performance of the portfolios.
        - The cumulative log returns for each portfolio and the market benchmark are plotted, with different line styles distinguishing between unadjusted and adjusted returns.

5. **Saving the Sharpe Ratios**:
    - **Conversion to DataFrame**: The list of Sharpe ratios is converted into a DataFrame for easier analysis and saving.
    - **Saving to CSV**: The resulting DataFrame containing Sharpe ratios for all fine-tuned models is saved to a CSV file for further analysis.
"""

# Step 14: Process each model's fine-tuned predictions
fine_tuned_predicted_directions = pd.read_csv('/content/merged_eval_df.csv')
fine_tuned_models = fine_tuned_predicted_directions['Model'].unique()
fine_tuned_all_sharpe_ratios = []

for model in fine_tuned_models:
    ew_l, ew_s, ew_ls, vw_l, vw_s, vw_ls, portfolio_returns, market_returns = process_model(
        model, fine_tuned_predicted_directions, returns_data_pivot, market_returns)
    adj_ew_l, adj_ew_s, adj_ew_ls, adj_vw_l, adj_vw_s, adj_vw_ls, adj_portfolio_returns, adj_market_returns = process_model(
        model, fine_tuned_predicted_directions, adjusted_returns_data_pivot, adjusted_market_returns)

    if ew_l is None or vw_l is None:
        continue

    model_sharpe_ratios = {
        'Model': model,
        'EW L': calculate_sharpe_ratio(ew_l),
        'EW S': calculate_sharpe_ratio(ew_s),
        'EW LS': calculate_sharpe_ratio(ew_ls),
        'VW L': calculate_sharpe_ratio(vw_l),
        'VW S': calculate_sharpe_ratio(vw_s),
        'VW LS': calculate_sharpe_ratio(vw_ls),
        'Market': calculate_sharpe_ratio(market_returns),
        # 'Adj EW L': calculate_sharpe_ratio(adj_ew_l),
        # 'Adj EW S': calculate_sharpe_ratio(adj_ew_s),
        # 'Adj EW LS': calculate_sharpe_ratio(adj_ew_ls),
        # 'Adj VW L': calculate_sharpe_ratio(adj_vw_l),
        # 'Adj VW S': calculate_sharpe_ratio(adj_vw_s),
        # 'Adj VW LS': calculate_sharpe_ratio(adj_vw_ls),
        # 'Adj Market': calculate_sharpe_ratio(adj_market_returns)
    }

    fine_tuned_all_sharpe_ratios.append(model_sharpe_ratios)

    # Plotting Cumulative Returns
    plt.figure(figsize=(12, 8))
    for label, returns in portfolio_returns.items():
        plt.plot((1 + returns).cumprod() - 1, label=label)
    # for label, returns in adj_portfolio_returns.items():
    #     plt.plot((1 + returns).cumprod() - 1, label=label, linestyle='--')
    plt.plot((1 + market_returns).cumprod() - 1, label='Market', linestyle=':')
    # plt.plot((1 + adj_market_returns).cumprod() - 1, label='Adj Market', linestyle='-.')
    plt.title(f'Cumulative Weekly Portfolio Returns Over Time for Fine-Tuned {model}')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Log Returns', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 15: Convert Sharpe ratios to DataFrame and save
fine_tuned_sharpe_ratios_df = pd.DataFrame(fine_tuned_all_sharpe_ratios)
print(fine_tuned_sharpe_ratios_df)

# Save the results to a CSV file
fine_tuned_sharpe_ratios_df.to_csv('/content/fine_tuned_sharpe_ratios.csv', index=False)

"""## Processing and Analyzing Un-Tuned Model Predictions for Risk and Return Performance

This section focuses on processing predictions from various models to calculate risk and return statistics, ultimately creating a detailed performance analysis.

### Key Steps in the Process

1. **Loading Predictions**:
    - **`un_tuned_predicted_directions`**: The DataFrame containing the predicted directions for each stock based on un-tuned models is loaded from a CSV file.
    - **`un_tuned_models`**: Unique model identifiers are extracted to iterate over each model’s predictions.

2. **Processing Each Model's Predictions**:
    - For each model, predictions are processed under two scenarios:
        1. **Without Transaction Costs**: Raw portfolio returns are calculated without accounting for transaction costs.
        2. **With Transaction Costs**: Portfolio returns are adjusted by applying transaction costs.
    - **`process_model()`**: This function calculates portfolio returns for both equally weighted (EW) and value weighted (VW) portfolios, including long-only, short-only, and long-short configurations.

3. **Calculating and Storing Sharpe Ratios**:
    - **`calculate_sharpe_ratio()`**: Sharpe ratios are calculated for each portfolio configuration (EW L, EW S, EW LS, VW L, VW S, VW LS).
    - **Performance Metrics**: For each portfolio type and scenario, the following metrics are stored:
        - **Return**: The mean return of the portfolio.
        - **Standard Deviation (Std)**: The volatility of the portfolio returns.
        - **Sharpe Ratio (SR)**: A measure of risk-adjusted return.

4. **Creating a Performance Summary**:
    - **`pivot_df`**: The collected results are pivoted into a multi-level DataFrame for better presentation, organizing metrics by model, scenario, and portfolio type.
    - **Styling and Saving**:
        - **Pandas Styling**: The DataFrame is styled for readability, with numeric formatting applied to the performance metrics.
        - **Saving Outputs**: The styled DataFrame is saved as an HTML file, and the underlying data is also saved as an Excel file.

"""

# Step 16: Process each model's predictions and calculate statistics
un_tuned_predicted_directions = pd.read_csv('/content/predicted_rolling_window_test_directions.csv')
un_tuned_models = un_tuned_predicted_directions['Model'].unique()
un_tuned_all_sharpe_ratios = []

results = {
    'Model': [],
    'Scenario': [],
    'Type': [],
    'Return': [],
    'Std': [],
    'SR': []
}

for model in un_tuned_models:
    for scenario, data, market_ret in [
        ('Without Transaction Cost', returns_data_pivot, market_returns),
        ('With Transaction Cost', adjusted_returns_data_pivot, adjusted_market_returns)
    ]:
        ew_l, ew_s, ew_ls, vw_l, vw_s, vw_ls, _, _ = process_model(
            model, un_tuned_predicted_directions, data, market_ret)

        model_sharpe_ratios = {
            'EW L': calculate_sharpe_ratio(ew_l),
            'EW S': calculate_sharpe_ratio(ew_s),
            'EW LS': calculate_sharpe_ratio(ew_ls),
            'VW L': calculate_sharpe_ratio(vw_l),
            'VW S': calculate_sharpe_ratio(vw_s),
            'VW LS': calculate_sharpe_ratio(vw_ls)
        }

        for port_type, returns_data in [
            ('EW L', ew_l),
            ('EW S', ew_s),
            ('EW LS', ew_ls),
            ('VW L', vw_l),
            ('VW S', vw_s),
            ('VW LS', vw_ls)
        ]:
            results['Model'].append(model)
            results['Scenario'].append(scenario)
            results['Type'].append(port_type)
            results['Return'].append(returns_data.mean())
            results['Std'].append(returns_data.std())
            results['SR'].append(model_sharpe_ratios[port_type])

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Pivot the DataFrame to get a multi-level index for better presentation
pivot_df = results_df.pivot_table(
    index=['Type', 'Scenario'],
    columns='Model',
    values=['Return', 'Std', 'SR']
)

# Reorder for the presentation similar to the image
pivot_df = pivot_df.swaplevel(0, 1).sort_index(level=0)
pivot_df.columns = [' '.join(col).strip() for col in pivot_df.columns.values]
pivot_df = pivot_df.reset_index()


# Display the DataFrame in a readable format with Pandas styling
styled_df = pivot_df.style.format({
    'Return': '{:.3f}',
    'Std': '{:.3f}',
    'SR': '{:.3f}'
}).set_caption("Risk and Return Performance for Pre-trained LLMs")

styled_df = styled_df.set_table_styles({
    'Scenario': [
        {'selector': 'th', 'props': [('font-size', '10pt'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ],
    'Model': [
        {'selector': 'th', 'props': [('font-size', '10pt'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ],
    'Type': [
        {'selector': 'th', 'props': [('font-size', '10pt'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ]
})

styled_df.to_html('/content/risk_and_return_performance_pre_trained.html')
pivot_df.to_excel('/content/risk_and_return_performance_pre_trained.xlsx', index=False)

"""## Processing and Analyzing Fine-Tuned Model Predictions for Risk and Return Performance

This section is focused on processing predictions from fine-tuned models to calculate risk and return statistics, providing a detailed performance analysis.

### Key Steps in the Process

1. **Loading Predictions**:
    - **`fine_tuned_predicted_directions`**: The DataFrame containing the predicted directions for each stock based on fine-tuned models is loaded from a CSV file.
    - **`fine_tuned_predicted_models`**: Unique model identifiers are extracted to iterate over each model’s predictions.

2. **Processing Each Model's Predictions**:
    - For each model, predictions are processed under two scenarios:
        1. **Without Transaction Costs**: Raw portfolio returns are calculated without accounting for transaction costs.
        2. **With Transaction Costs**: Portfolio returns are adjusted by applying transaction costs.
    - **`process_model()`**: This function calculates portfolio returns for both equally weighted (EW) and value weighted (VW) portfolios, including long-only, short-only, and long-short configurations.

3. **Calculating and Storing Sharpe Ratios**:
    - **`calculate_sharpe_ratio()`**: Sharpe ratios are calculated for each portfolio configuration (EW L, EW S, EW LS, VW L, VW S, VW LS).
    - **Performance Metrics**: For each portfolio type and scenario, the following metrics are stored:
        - **Return**: The mean return of the portfolio.
        - **Standard Deviation (Std)**: The volatility of the portfolio returns.
        - **Sharpe Ratio (SR)**: A measure of risk-adjusted return.

4. **Creating a Performance Summary**:
    - **`pivot_df`**: The collected results are pivoted into a multi-level DataFrame for better presentation, organizing metrics by model, scenario, and portfolio type.
    - **Styling and Saving**:
        - **Pandas Styling**: The DataFrame is styled for readability, with numeric formatting applied to the performance metrics.
        - **Saving Outputs**: The styled DataFrame is saved as an HTML file, and the underlying data is also saved as an Excel file.
"""

# Step 17: Process each model's predictions and calculate statistics
fine_tuned_predicted_directions = pd.read_csv('/content/merged_fine_tuned_results_df.csv')
fine_tuned_predicted_models = fine_tuned_predicted_directions['Model'].unique()
un_tuned_all_sharpe_ratios = []

results = {
    'Model': [],
    'Scenario': [],
    'Type': [],
    'Return': [],
    'Std': [],
    'SR': []
}

for model in fine_tuned_predicted_models:
    for scenario, data, market_ret in [
        ('Without Transaction Cost', returns_data_pivot, market_returns),
        ('With Transaction Cost', adjusted_returns_data_pivot, adjusted_market_returns)
    ]:
        ew_l, ew_s, ew_ls, vw_l, vw_s, vw_ls, _, _ = process_model(
            model, fine_tuned_predicted_directions, data, market_ret)

        model_sharpe_ratios = {
            'EW L': calculate_sharpe_ratio(ew_l),
            'EW S': calculate_sharpe_ratio(ew_s),
            'EW LS': calculate_sharpe_ratio(ew_ls),
            'VW L': calculate_sharpe_ratio(vw_l),
            'VW S': calculate_sharpe_ratio(vw_s),
            'VW LS': calculate_sharpe_ratio(vw_ls)
        }

        for port_type, returns_data in [
            ('EW L', ew_l),
            ('EW S', ew_s),
            ('EW LS', ew_ls),
            ('VW L', vw_l),
            ('VW S', vw_s),
            ('VW LS', vw_ls)
        ]:
            results['Model'].append(model)
            results['Scenario'].append(scenario)
            results['Type'].append(port_type)
            results['Return'].append(returns_data.mean())
            results['Std'].append(returns_data.std())
            results['SR'].append(model_sharpe_ratios[port_type])

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Pivot the DataFrame to get a multi-level index for better presentation
pivot_df = results_df.pivot_table(
    index=['Type', 'Scenario'],
    columns='Model',
    values=['Return', 'Std', 'SR']
)

# Reorder for the presentation similar to the image
pivot_df = pivot_df.swaplevel(0, 1).sort_index(level=0)
pivot_df.columns = [' '.join(col).strip() for col in pivot_df.columns.values]
pivot_df = pivot_df.reset_index()


# Display the DataFrame in a readable format with Pandas styling
styled_df = pivot_df.style.format({
    'Return': '{:.3f}',
    'Std': '{:.3f}',
    'SR': '{:.3f}'
}).set_caption("Risk and Return Performance for Fine-trained LLMs")

styled_df = styled_df.set_table_styles({
    'Scenario': [
        {'selector': 'th', 'props': [('font-size', '10pt'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ],
    'Model': [
        {'selector': 'th', 'props': [('font-size', '10pt'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ],
    'Type': [
        {'selector': 'th', 'props': [('font-size', '10pt'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ]
})

styled_df.to_html('/content/risk_and_return_performance_fine_tuned.html')
pivot_df.to_excel('/content/risk_and_return_performance_fine_tuned.xlsx', index=False)