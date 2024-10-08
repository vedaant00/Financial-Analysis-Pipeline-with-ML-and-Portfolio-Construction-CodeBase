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
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

"""## Rolling Window Analysis with Embeddings

This section explains the implementation of a rolling window analysis using embeddings generated by various models. The objective is to evaluate the performance of these models in predicting market direction (up or down) based on historical data.

### Key Functions and Workflow

1. **`rolling_window_analysis_with_embeddings`**:
   - This function performs logistic regression on the embedding data, evaluating the model's performance using metrics such as accuracy, precision, recall, and F1 score.
   - It also generates a confusion matrix and stores the predicted directions.

2. **Rolling Window Analysis Process**:
   - The analysis is performed for each `Permco` (company identifier) by combining the in-sample and out-sample datasets.
   - The data is split into a rolling window, where 10 years of data are used for training, and the next year is used for testing.
   - The rolling window continues across all years available in the dataset, evaluating the model's performance in predicting the market direction for each year.

3. **Iterating Over Models and Years**:
   - The script iterates over different models (e.g., BERT, RoBERTa) and years, applying the rolling window methodology.
   - For each iteration, the script checks for the existence of embeddings and skips the analysis if embeddings are missing.

4. **Result Aggregation**:
   - The results of the rolling window analysis are stored in a dictionary structure, organized by `Permco`, year, and model.
   - Additionally, the predicted directions are compiled into a DataFrame for further analysis or visualization.

### Execution Summary

- **Data Preparation**:
  - The combined dataset for each `Permco` is prepared by concatenating in-sample and out-sample data. The dataset is then sorted by date and split into training and testing periods.
  
- **Rolling Window Application**:
  - For each `Permco` and model, the rolling window analysis is conducted across the available years.
  - The function ensures that each training and testing set is of sufficient size, skipping any iteration where the data is insufficient.

- **Result Storage**:
  - The analysis results, including accuracy, precision, recall, F1 score, and confusion matrices, are stored in a dictionary.
  - Predicted directions are saved in a CSV file for further examination.

### Output

- **`predicted_rolling_window_test_directions.csv`**:
  - This file contains the predicted directions for each company and model, along with the true directions, allowing for a detailed comparison and analysis.
"""

def rolling_window_analysis_with_embeddings(train_data, test_data, model_key):
    results = []

    train_embeddings = np.array(train_data[f'{model_key}_avg_embedding'].tolist())
    test_embeddings = np.array(test_data[f'{model_key}_avg_embedding'].tolist())

    train_labels = train_data['Direction'].values
    test_labels = test_data['Direction'].values

    model = LogisticRegression()
    model.fit(train_embeddings, train_labels)
    predictions = model.predict(test_embeddings)

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')
    cm = confusion_matrix(test_labels, predictions)

    results.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predicted_direction': predictions.tolist()
    })

    return results

rolling_window_results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
predicted_directions_df_list = []

permcos = in_sample_data['Permco'].unique()
for permco in permcos:
    print(f"Performing rolling window analysis for {permco}...")

    try:
        combined_data = pd.concat([in_sample_data[in_sample_data['Permco'] == permco],
                                   out_sample_data[out_sample_data['Permco'] == permco]])
    except FileNotFoundError:
        print(f"Skipping {permco} due to missing file.")
        continue

    combined_data['Start_Date'] = pd.to_datetime(combined_data['Start_Date'])

    for model_key in model_name_mapping.keys():
        if f'{model_key}_avg_embedding' not in combined_data.columns:
            print(f"Skipping {permco} for model {model_key} due to missing embeddings.")
            continue

        combined_data[f'{model_key}_avg_embedding'] = combined_data[f'{model_key}_avg_embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        unique_years = sorted(combined_data['Start_Date'].dt.year.unique())
        print(f"Unique years for {permco} and model {model_key}: {unique_years}")

        for i in range(10, len(unique_years) - 1):
            train_years = unique_years[i-10:i]
            test_year = unique_years[i]

            print(f"Training years for {permco}, model {model_key}, test year {test_year}: {train_years}")

            rolling_train_data = combined_data[combined_data['Start_Date'].dt.year.isin(train_years)]
            rolling_test_data = combined_data[combined_data['Start_Date'].dt.year == test_year]

            print(f"Rolling train data length: {len(rolling_train_data)}")
            print(f"Rolling test data length: {len(rolling_test_data)}")

            if rolling_train_data.empty or rolling_test_data.empty:
                print(f"Skipping year {test_year} for {permco} due to empty rolling train or test data.")
                continue

            result = rolling_window_analysis_with_embeddings(rolling_train_data, rolling_test_data, model_key)[0]

            rolling_window_results_dict[permco][test_year][model_key] = result

            if len(result['predicted_direction']) != len(rolling_test_data):
                print(f"Length mismatch: {len(result['predicted_direction'])} vs {len(rolling_test_data)}")
                continue

            predicted_directions_df = pd.DataFrame({
                'Permco': [permco] * len(rolling_test_data),
                'Date': rolling_test_data['Start_Date'].values,
                'Ticker': rolling_test_data['Ticker'].values,
                'True_Direction': rolling_test_data['Direction'].values,
                'Predicted_Direction': result['predicted_direction'],
                'Model': [model_key] * len(rolling_test_data)
            })

            predicted_directions_df_list.append(predicted_directions_df)

if predicted_directions_df_list:
    predicted_directions_df = pd.concat(predicted_directions_df_list, ignore_index=True)
    predicted_directions_df.to_csv('predicted_rolling_window_test_directions.csv', index=False)
else:
    print("No predicted directions to concatenate.")

"""## Flattening Nested Dictionaries and Converting Rolling Window Results to DataFrame

This section details the process of flattening nested dictionaries and converting the rolling window analysis results into a structured DataFrame for easier analysis and export.

### Key Functions and Workflow

1. **`flatten_dict`**:
   - This recursive function is designed to flatten a nested dictionary.
   - It takes a dictionary as input and returns a new dictionary where nested keys are concatenated with a separator (e.g., an underscore) to create a flat structure.
   - This is particularly useful for handling the complex, nested structures that arise from storing multiple metrics in a hierarchical format.

2. **`rolling_results_to_df`**:
   - This function processes the `rolling_window_results_dict`, which contains the results of the rolling window analysis.
   - It iterates over the dictionary, flattening the metrics and combining them with the `permco`, `year`, and `model` identifiers.
   - The results are then stored in a list of records, which is converted into a Pandas DataFrame.

3. **Exporting Results**:
   - The DataFrame containing the flattened rolling window results is saved to a CSV file for further analysis or reporting.

### Execution Summary

- **Flattening Nested Metrics**:
  - The `flatten_dict` function is applied to each set of metrics within the `rolling_window_results_dict`.
  - This ensures that all metrics, regardless of their original depth in the nested structure, are accessible as individual columns in the resulting DataFrame.

- **Converting to DataFrame**:
  - The `rolling_results_to_df` function systematically converts the entire results dictionary into a DataFrame, where each row corresponds to a unique combination of `permco`, `year`, and `model`.
  - The DataFrame provides a structured and flat representation of the rolling window results, making it easy to analyze the data across different dimensions.

- **Result Storage**:
  - The final DataFrame is saved as a CSV file, `rolling_window_test_results.csv`, which can be used for further data exploration, visualization, or integration into reports.

### Output

- **`rolling_window_test_results.csv`**:
  - This file contains the flattened and structured results of the rolling window analysis, with columns representing various metrics, `permco`, `year`, and `model`.
  - The flat format makes it easier to perform cross-comparisons, aggregations, and other forms of analysis on the rolling window results.
"""

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def rolling_results_to_df(rolling_window_results_dict):
    records = []
    for permco, years in rolling_window_results_dict.items():
        for year, models in years.items():
            for model, metrics in models.items():
                flattened_metrics = flatten_dict(metrics)
                record = {'permco': permco, 'year': year, 'model': model}
                record.update(flattened_metrics)
                records.append(record)
    return pd.DataFrame(records)

rolling_results_df = rolling_results_to_df(rolling_window_results_dict)
rolling_results_df.to_csv('/content/datasets/rolling_window_test_results.csv', index=False)

"""## Analyzing Model Performance (Rolling Window on Base Models): Average Accuracies, Yearwise Metrics, Visualisations and Best Model Identification

This section outlines the process of evaluating and visualizing the performance of various models used in rolling window analysis. The focus is on plotting average accuracies across years, comparing yearwise metrics for different models, and identifying the best-performing models both yearwise and tickerwise.

### Key Functions and Workflow

1. **Plotting Average Accuracies**:
   - **`plot_average_accuracies(results_dict)`**:
     - This function calculates and plots the average accuracies of different models across years.
     - For each year, it aggregates the accuracies from all `permco`s and computes the mean accuracy for each model.
     - The results are visualized using line plots, with each model represented by a distinct line, allowing for a clear comparison of model performance over time.

2. **Yearwise Metrics for Each Model**:
   - **`plot_all_metrics_each_model(results_dict)`**:
     - This function focuses on plotting multiple performance metrics (accuracy, precision, recall, and F1-score) for each model, year by year.
     - It provides a detailed view of how each model performs across different metrics, making it easier to identify trends and performance stability over time.

   - **`plot_all_metrics_each_model_ticker(results_dict)`**:
     - Similar to the previous function, but tailored to analyze the performance metrics for each `permco` separately.
     - This function plots yearwise metrics for each model within a specific `permco`, offering a more granular view of the model's performance for individual companies.

3. **Identifying the Best Model**:
   - **`find_best_model(results_dict)`**:
     - This function identifies the best-performing model for each year by comparing the average accuracy across all models.
     - It generates a DataFrame that highlights which model performed best each year and also determines the overall best model across all years.

   - **`plot_results_table(df, overall_best_model)`**:
     - Once the best models are identified, this function visualizes the results in a tabular format, making it easy to understand which models excelled in which years.
     - The table is displayed along with the overall best model, providing a clear summary of model performance.

   - **`plot_best_model_by_year(df)`**:
     - This function creates a bar plot showing the frequency of each model being selected as the best model across different years.
     - It gives a quick visual summary of which models are consistently performing well over time.

4. **Best Model by Ticker (Company)**:
   - **`find_best_model_tickerwise(results_dict)`**:
     - This function identifies the best model for each `permco` by analyzing model performance across different years.
     - It produces a DataFrame that shows the best model for each company and also determines the overall best model tickerwise.

   - **`plot_best_model_by_ticker(df)`**:
     - This function visualizes the best model for each company using a pie chart.
     - The pie chart provides an intuitive understanding of which models are the most effective across the majority of companies.

### Execution Summary

- **Average Accuracies**:
  - The average accuracies of models are plotted across different years, highlighting the overall trends and stability of each model's performance over time.

- **Yearwise Performance Metrics**:
  - Detailed metrics (accuracy, precision, recall, F1-score) are plotted for each model, offering a comprehensive view of their strengths and weaknesses year by year.

- **Best Model Identification**:
  - The best models are identified both on a yearly basis and across companies, with results visualized in tables and plots for easy interpretation.

- **Results Visualization**:
  - The plots and tables provide clear and actionable insights into model performance, helping in the selection of the most robust models for different scenarios.

"""

def plot_average_accuracies(results_dict):
    years = sorted(set(year for permco in results_dict for year in results_dict[permco]))
    avg_accuracies = {model: [] for model in model_name_mapping.keys()}

    for year in years:
        for model_name in model_name_mapping.keys():
            accuracies = [results_dict[permco][year][model_name]['accuracy']
                          for permco in results_dict
                          if year in results_dict[permco] and model_name in results_dict[permco][year]]
            avg_accuracies[model_name].append(np.mean(accuracies) if accuracies else np.nan)

    plt.figure(figsize=(15, 8))
    for model_name, accuracies in avg_accuracies.items():
        plt.plot(years, accuracies, label=model_name, marker='o')

    plt.xlabel('Year')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracies of Models Across the Years')
    plt.legend()
    plt.show()

def plot_all_metrics_each_model(results_dict):
    for model_name in model_name_mapping.keys():
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        years = sorted(set(year for permco in results_dict for year in results_dict[permco]))
        avg_metrics = {metric: [] for metric in metrics}

        for year in years:
            for metric in metrics:
                values = [results_dict[permco][year][model_name][metric]
                          for permco in results_dict
                          if year in results_dict[permco] and model_name in results_dict[permco][year]]
                avg_metrics[metric].append(np.mean(values) if values else np.nan)

        plt.figure(figsize=(15, 8))
        for metric, values in avg_metrics.items():
            plt.plot(years, values, label=metric, marker='o')

        plt.xlabel('Year')
        plt.ylabel('Score')
        plt.title(f'Yearwise Performance Metrics for {model_name.upper()}')
        plt.legend()
        plt.show()

def plot_all_metrics_each_model_ticker(results_dict):
    for permco in results_dict:
        for model_name in model_name_mapping.keys():
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            years = sorted(results_dict[permco].keys())
            metric_values = {metric: [] for metric in metrics}

            for year in years:
                for metric in metrics:
                    if model_name in results_dict[permco][year]:
                        metric_values[metric].append(results_dict[permco][year][model_name][metric])
                    else:
                        metric_values[metric].append(np.nan)

            plt.figure(figsize=(15, 8))
            for metric, values in metric_values.items():
                plt.plot(years, values, label=metric, marker='o')

            plt.xlabel('Year')
            plt.ylabel('Score')
            plt.title(f'Yearwise Performance Metrics for {model_name.upper()} ({permco})')
            plt.legend()
            plt.show()

def find_best_model(results_dict):
    all_years = sorted({year for permco in results_dict for year in results_dict[permco]})
    all_models = sorted({model for permco in results_dict for year in results_dict[permco] for model in results_dict[permco][year]})

    year_model_accuracy = {year: {model: [] for model in all_models} for year in all_years}

    for permco in results_dict:
        for year in results_dict[permco]:
            for model in results_dict[permco][year]:
                accuracy = results_dict[permco][year][model].get('accuracy', np.nan)
                year_model_accuracy[year][model].append(accuracy)

    year_model_avg_accuracy = {year: {model: np.nanmean(year_model_accuracy[year][model]) for model in all_models} for year in all_years}

    year_df = pd.DataFrame(year_model_avg_accuracy).T
    year_df['Best_Model'] = year_df.idxmax(axis=1)

    overall_best_model_yearwise = year_df[all_models].mean().idxmax()

    return year_df, overall_best_model_yearwise

def plot_results_table(df, overall_best_model):
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('tight')
    ax.axis('off')
    table = pd.plotting.table(ax, df, loc='center', cellLoc='center', colWidths=[0.2] * len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 1.5)
    plt.title(f'Best Models by Year\nOverall Best Model: {overall_best_model}', fontsize=18)
    plt.show()

def plot_best_model_by_year(df):
    best_model_counts = df['Best_Model'].value_counts()
    plt.figure(figsize=(10, 6))
    best_model_counts.plot(kind='bar', color=['blue', 'orange', 'green', 'red', 'yellow'])
    plt.title('Best Model by Year (ignoring the company)', fontsize=16, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Number of Times Selected as Best')
    plt.show()

def find_best_model_tickerwise(results_dict):
    permcos = sorted(results_dict.keys())
    all_models = sorted({model for permco in results_dict for year in results_dict[permco] for model in results_dict[permco][year]})

    ticker_model_accuracy = {permco: {model: [] for model in all_models} for permco in permcos}

    for permco in results_dict:
        for year in results_dict[permco]:
            for model in results_dict[permco][year]:
                accuracy = results_dict[permco][year][model].get('accuracy', np.nan)
                ticker_model_accuracy[permco][model].append(accuracy)

    ticker_model_avg_accuracy = {permco: {model: np.nanmean(ticker_model_accuracy[permco][model]) for model in all_models} for permco in permcos}

    ticker_df = pd.DataFrame(ticker_model_avg_accuracy).T
    ticker_df['Best_Model'] = ticker_df.idxmax(axis=1)

    overall_best_model_tickerwise = ticker_df[all_models].mean().idxmax()

    return ticker_df, overall_best_model_tickerwise

def plot_best_model_by_ticker(df):
    best_model_counts = df['Best_Model'].value_counts()
    plt.figure(figsize=(8, 8))
    best_model_counts.plot(kind='pie', autopct='%1.1f%%', colors=['blue', 'orange', 'green', 'red', 'yellow'])
    plt.title('Best Model by Majority Company', fontsize=16, fontweight='bold')
    plt.ylabel('')
    plt.show()

# Generate plots for rolling window analysis
plot_average_accuracies(rolling_window_results_dict)
plot_all_metrics_each_model(rolling_window_results_dict)
plot_all_metrics_each_model_ticker(rolling_window_results_dict)

# Find best models by year and ticker
year_df, overall_best_model_yearwise = find_best_model(rolling_window_results_dict)
plot_results_table(year_df, overall_best_model_yearwise)
plot_best_model_by_year(year_df)

ticker_df, overall_best_model_tickerwise = find_best_model_tickerwise(rolling_window_results_dict)
plot_best_model_by_ticker(ticker_df)

## Calculating the MSE and RMSE for Rolling Window (Embeddings + Logistic Regression) Predictions
data = pd.read_csv('/content/predicted_rolling_window_test_directions.csv')

# Initialize dictionaries to store MAE and MSE
rmse_results = {}
mse_results = {}

# Get unique models from the data
models = data['Model'].unique()

# Calculate MSE and RMSE for each model
for model in models:
    model_data = data[data['Model'] == model]
    true_values = model_data['True_Direction']
    predicted_values = model_data['Predicted_Direction']

    mse = mean_squared_error(true_values, predicted_values)

    mse_results[model] = mse
    rmse = np.sqrt(mse)
    rmse_results[model] = rmse

# Display the results
print("\nMSE for each model:")
for model, mse in mse_results.items():
    print(f"{model}: {mse}")

print("\nRMSE for each model:")
for model, rmse in rmse_results.items():
    print(f"{model}: {rmse}")

 ## Calculating the MSE and RMSE for Fine-Tuned Predictions   
data = pd.read_csv('/content/merged_fine_tuned_results_df.csv')

# Initialize dictionaries to store MAE and MSE
mae_results = {}
mse_results = {}
rmse_results = {}

# Get unique models from the data
models = data['Model'].unique()

# Calculate MAE and MSE and RMSE for each model
for model in models:
    model_data = data[data['Model'] == model]
    true_values = model_data['Direction']
    predicted_values = model_data['Predicted_Direction']

    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)

    mae_results[model] = mae
    mse_results[model] = mse
    rmse_results[model] = np.sqrt(mse)

print("\nMSE for each model:")
for model, mse in mse_results.items():
    print(f"{model}: {mse}")

print("\nRMSE for each model:")
for model, rmse in rmse_results.items():
    print(f"{model}: {rmse}")