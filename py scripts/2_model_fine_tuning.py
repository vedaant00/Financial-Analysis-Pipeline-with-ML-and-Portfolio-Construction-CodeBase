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

"""## Custom Dataset Creation and Embedding Generation

This section involves creating a custom dataset class and generating embeddings for text data using various pretrained models.

### Custom Dataset Class

- **CustomDataset Class**:
  - A custom dataset class is defined to handle the encoded input data and corresponding labels.
  - The class provides the necessary methods (`__getitem__` and `__len__`) to make it compatible with PyTorch's DataLoader, allowing for efficient batching and shuffling during training.

### Data Preparation and Embedding Generation

- **Data Preparation**:
  - The `prepare_data` function tokenizes the text data (headlines) using a specified tokenizer, ensuring truncation and padding to a fixed maximum length.
  - The labels are extracted from the DataFrame and prepared for model input.

- **Embedding Generation**:
  - The `generate_embeddings` function computes two types of embeddings for each text input:
    - **CLS Embedding**: The embedding of the `[CLS]` token, which is commonly used for classification tasks.
    - **Average Embedding**: The average of all token embeddings in the sequence, providing a more holistic representation of the input.
  - The embeddings are generated in batches for efficiency and stored back into the DataFrame.

### Implementation Steps

1. **Device Setup**:
   - The computation is configured to use a GPU if available; otherwise, it defaults to CPU.

2. **Model Setup**:
   - Pretrained models like BERT, RoBERTa, DistilBERT, DistilRoBERTa, and FinBERT are loaded, and their configurations are adjusted to output hidden states (required for extracting embeddings).

3. **Embedding Computation**:
   - For each model, the text data from both in-sample and out-sample datasets is processed to generate embeddings.
   - These embeddings are stored within the respective DataFrames.

4. **Saving the Embeddings**:
   - The generated embeddings are saved as CSV files for each company identified by `Permco`, with separate files for training (in-sample) and testing (out-sample) data.
"""

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def prepare_data(df, tokenizer, max_length=512):
    encodings = tokenizer(list(df['Headline'].values), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    labels = list(df['Direction'].values)
    return encodings, labels

def generate_embeddings(df, tokenizer, model, device, model_name, batch_size=16):
    model.eval()  # Set model to evaluation mode
    embeddings = []
    avg_embeddings = []

    for i in range(0, len(df), batch_size):
        batch_texts = df['Headline'].values[i:i + batch_size]
        encodings = tokenizer(list(batch_texts), truncation=True, padding=True, max_length=512, return_tensors='pt')
        encodings = {key: val.to(device) for key, val in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            hidden_states = outputs.hidden_states[-1]
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy()  # Use the CLS token embedding
            avg_embeddings_batch = hidden_states.mean(dim=1).cpu().numpy()  # Average embeddings

        embeddings.extend(cls_embeddings)
        avg_embeddings.extend(avg_embeddings_batch)

    df[f'{model_name}_embedding'] = [cls_emb.tolist() for cls_emb in embeddings]
    df[f'{model_name}_avg_embedding'] = [avg_emb.tolist() for avg_emb in avg_embeddings]

    return df

# Generate embeddings for in-sample and out-sample data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_key, model_name in model_name_mapping.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    model.config.output_hidden_states = True
    model.to(device)

    in_sample_data = generate_embeddings(in_sample_data, tokenizer, model, device, model_key)
    out_sample_data = generate_embeddings(out_sample_data, tokenizer, model, device, model_key)

# # Save the generated embeddings
for permco in in_sample_data['Permco'].unique():
    train_data = in_sample_data[in_sample_data['Permco'] == permco]
    test_data = out_sample_data[out_sample_data['Permco'] == permco]
    os.makedirs(f'/content/datasets/{permco}', exist_ok=True)
    train_data.to_csv(f'/content/datasets/{permco}/train_embeddings.csv', index=False)
    test_data.to_csv(f'/content/datasets/{permco}/test_embeddings.csv', index=False)

"""## Model Training and Evaluation with Metrics Logging and Plotting

This section covers the training and evaluation of sequence classification models using text embeddings. It includes custom callback functions for logging training metrics, as well as functions for preparing data, training models, and visualizing results.

### Custom Callback for Metrics Logging

- **MetricsCallback Class**:
  - A custom callback class is defined to log training loss and evaluation metrics (e.g., accuracy) during the training process.
  - The callback captures the loss at each logging step and the evaluation accuracy at each evaluation step.

### Metric Computation and Evaluation

- **`compute_metrics` Function**:
  - This function computes accuracy, precision, recall, and F1-score based on the model's predictions.
  - It uses `accuracy_score` and `precision_recall_fscore_support` from `sklearn` to calculate these metrics.

- **`evaluate_and_get_logits` Function**:
  - This function evaluates the model on a given dataset and retrieves the logits (model output before applying the softmax function) and the true labels.
  - It uses a DataLoader to process the dataset in batches for efficiency.

### Model Training and Fine-Tuning

- **`train_and_evaluate_model` Function**:
  - This function trains and evaluates a model using the provided training and evaluation datasets.
  - It uses the `Trainer` class from the `transformers` library to handle the training loop, evaluation, and logging.
  - The function also saves the trained model's predictions and confusion matrix to CSV files.

- **Training Setup**:
  - The training process is set up with specific arguments, including batch size, number of epochs, and evaluation strategy.
  - Models are trained and evaluated for each company (`Permco`) in the dataset using different pretrained models (e.g., BERT, RoBERTa).

### Plotting Training Metrics

- **`plot_training_metrics` Function**:
  - This function visualizes the training loss over time by plotting the loss at each training step.
  - The plot helps in understanding the training process and detecting any potential issues such as overfitting.

### Execution and Results Storage

- **Training and Evaluation Loop**:
  - The code iterates over all companies in the dataset, training a separate model for each one.
  - For each company and model, the training and evaluation results are stored in a dictionary, which includes metrics like accuracy, precision, recall, and F1-score.
  - The results are saved separately for each year of the evaluation data.

- **Handling CUDA Memory Issues**:
  - The code includes a mechanism to handle CUDA out-of-memory errors by skipping the problematic company and clearing the GPU memory cache.
"""

class MetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_logs = []
        self.eval_metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.train_logs.append({
                'step': state.global_step,
                'loss': logs.get('loss', None)
            })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self.eval_metrics.append({
                'eval_step': state.global_step,
                'eval_accuracy': metrics.get('eval_accuracy', None)
            })

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_and_get_logits(model, dataset, device):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()
            all_logits.extend(logits)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_logits), np.array(all_labels)

def train_and_evaluate_model(train_df, eval_df, model_name, tokenizer_name, permco, drive_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_encodings, train_labels = prepare_data(train_df, tokenizer)
    eval_encodings, eval_labels = prepare_data(eval_df, tokenizer)

    train_dataset = CustomDataset(train_encodings, train_labels)
    eval_dataset = CustomDataset(eval_encodings, eval_labels)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    model.config.output_hidden_states = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = f'{drive_path}/{permco}/{model_name}'
    os.makedirs(output_dir, exist_ok=True)

    metrics_callback = MetricsCallback()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        logging_dir=f'{output_dir}/logs',
        logging_steps=1,
        evaluation_strategy='epoch',
        save_strategy='no',  # Disable saving checkpoints
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )

    trainer.train()
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)

    eval_logits, eval_labels = evaluate_and_get_logits(model, eval_dataset, device)
    confusion_mat = confusion_matrix(eval_labels, np.argmax(eval_logits, axis=-1))

    eval_df['Predicted_Direction'] = np.argmax(eval_logits, axis=-1)
    eval_df.to_csv(f'{output_dir}/fine_tuned_predictions.csv', index=False)

    # tokenizer.save_pretrained(f'{output_dir}/tokenizer')
    # model.save_pretrained(f'{output_dir}/model')

    return eval_results, model, tokenizer, device, eval_logits, confusion_mat, metrics_callback

def plot_training_metrics(metrics_callback):
    train_steps = [log['step'] for log in metrics_callback.train_logs if 'step' in log]
    train_loss = [log['loss'] for log in metrics_callback.train_logs if 'loss' in log]

    plt.figure(figsize=(12, 6))
    if train_steps and train_loss:
        plt.plot(train_steps, train_loss, label='Training Loss', linestyle='-')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

results_dict = {}
drive_path = '/content/drive/MyDrive/models'

permcos = in_sample_data['Permco'].unique()
for permco in permcos:
    print(f"Training and saving model for {permco}...")
    company_train_df = in_sample_data[in_sample_data['Permco'] == permco]
    company_test_df = out_sample_data[out_sample_data['Permco'] == permco]

    if company_train_df.empty or company_test_df.empty:
        continue

    try:
        for model_key, model_name in model_name_mapping.items():
            eval_results, model, tokenizer, device, eval_logits, confusion_mat, metrics_callback = train_and_evaluate_model(
                company_train_df, company_test_df, model_name, model_name, permco, drive_path)

            plot_training_metrics(metrics_callback)

            if permco not in results_dict:
                results_dict[permco] = {}

            unique_years = sorted(company_test_df['Start_Date'].dt.year.unique())
            for year in unique_years:
                year_data = company_test_df[company_test_df['Start_Date'].dt.year == year]
                if year not in results_dict[permco]:
                    results_dict[permco][year] = {}

                results_dict[permco][year][model_key] = {
                    'accuracy': eval_results['eval_accuracy'],
                    'precision': eval_results['eval_precision'],
                    'recall': eval_results['eval_recall'],
                    'f1': eval_results['eval_f1'],
                    'permco': permco,
                    'ticker': company_train_df['Ticker'].unique()[0],
                    'confusion_matrix': confusion_mat
                }

    except RuntimeError as e:
        if 'CUDA out' in str(e):
            print(f"Skipping {permco} due to CUDA out of memory error.")
            torch.cuda.empty_cache()
            continue
        else:
            raise e

"""## Merging Fine-Tuned Model Predictions

This section describes the process of merging prediction results from fine-tuned models across different companies (`Permco`) and models. The merged dataset is then saved for further analysis.

### Steps Involved

1. **Directory Setup**:
   - The base directory is defined, where the models are stored in a structured format under directories named after `Permco` values. Each `Permco` directory contains subdirectories for different models.

2. **Data Aggregation**:
   - The code iterates through each `Permco` directory and then through each model directory within it.
   - For each model, it checks if the `fine_tuned_predictions.csv` file exists, which contains the prediction results.

3. **Data Loading and Augmentation**:
   - If the predictions file is found, it is loaded into a DataFrame.
   - Additional columns are added to the DataFrame to indicate the corresponding `Permco` and model used for generating the predictions.

4. **Data Concatenation**:
   - All DataFrames from the different models and `Permco` directories are concatenated into a single DataFrame.

5. **Saving the Merged Dataset**:
   - The final merged DataFrame is saved as a CSV file for further analysis or reporting.

"""

base_dir = '/content/drive/MyDrive/models'

dfs = []

for permco_dir in os.listdir(base_dir):
    permco_path = os.path.join(base_dir, permco_dir)
    if os.path.isdir(permco_path):
        for model_dir in os.listdir(permco_path):
            model_path = os.path.join(permco_path, model_dir)
            eval_df_path = os.path.join(model_path, 'fine_tuned_predictions.csv')
            if os.path.exists(eval_df_path):
                # Load the eval_df file and append it to the list
                df = pd.read_csv(eval_df_path)
                df['Permco'] = permco_dir  # Add a column to identify the permco
                df['Model'] = model_dir  # Add a column to identify the model
                dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv('/content/merged_fine_tuned_results_df.csv', index=False)
print("Merged merged_fine_tuned_results_df saved to /content/merged_fine_tuned_results_df.csv")

"""## ## Analyzing Model Performance (Fine-Tuned): Average Accuracies, Yearwise Metrics, Visualisations and Best Model Identification

This section outlines the procedures for evaluating and visualizing the performance of fine-tuned models across different companies and years. The focus is on analyzing metrics such as accuracy, precision, recall, and F1 score, along with generating confusion matrices.

### Key Functions and Their Purposes

1. **`plot_confusion_matrix`**:
   - Plots a confusion matrix for a given `Permco`, model, and year.

2. **`plot_confusion_matrices`**:
   - Iterates through the results dictionary to generate and display confusion matrices for each model and year.

3. **`plot_yearwise_metrics_tickers`**:
   - Plots the year-wise performance metrics (accuracy, precision, recall, F1) for each model across different `Permcos`.

4. **`plot_yearwise_metrics_avg`**:
   - Averages the performance metrics across all `Permcos` and plots them year-wise for each model.

5. **`find_best_model`**:
   - Determines the best-performing model for each year by comparing accuracy metrics across all models.

6. **`plot_results_table`**:
   - Displays a summary table of the best models by year, highlighting the overall best model.

7. **`plot_best_model_by_year`**:
   - Visualizes the frequency of each model being selected as the best model by year.

8. **`find_best_model_tickerwise`**:
   - Identifies the best model for each `Permco` based on the average accuracy across all years.

9. **`plot_best_model_by_ticker`**:
   - Visualizes the distribution of the best models across all `Permcos` using a pie chart.

### Summary of Plots and Analysis

- **Year-wise Metrics by Ticker**: Shows the performance of each model across different years for each company.
- **Year-wise Average Metrics**: Aggregates the metrics across companies to provide an overall performance trend.
- **Best Model by Year**: Identifies which model performs the best for each year and presents a summary in a table format.
- **Best Model by Ticker**: Identifies the best-performing model for each company and visualizes it in a pie chart.

### Execution and Visualization

- The above functions are executed to generate plots and tables that provide insights into the performance of various models. The results help in identifying trends, comparing models, and determining the most effective models for different scenarios.

- **Confusion Matrices**: Detailed matrices are plotted to give a clear picture of the classification accuracy for each model.

- **Performance Metrics**: The metrics are plotted both in aggregate and individually to showcase the strengths and weaknesses of each model.
"""

def plot_confusion_matrix(cm, model_name, permco, year):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {permco} ({model_name}) - {year}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_confusion_matrices(results_dict):
    for permco, years in results_dict.items():
        for year, models in years.items():
            for model_name, metrics in models.items():
                if 'confusion_matrix' in metrics:
                    plot_confusion_matrix(metrics['confusion_matrix'], model_name, permco, year)

def plot_yearwise_metrics_tickers(results_dict):
    model_metrics = {model_name: {} for model_name in model_name_mapping.keys()}

    for permco, years in results_dict.items():
        for year, models in years.items():
            for model_name, metrics in models.items():
                if year not in model_metrics[model_name]:
                    model_metrics[model_name][year] = {metric: [] for metric in metrics.keys() if metric != 'confusion_matrix'}
                for metric, value in metrics.items():
                    if metric != 'confusion_matrix':
                        try:
                            model_metrics[model_name][year][metric].append(float(value))
                        except ValueError:
                            continue

    for permco in results_dict.keys():
        for model_name in model_name_mapping.keys():
            plt.figure(figsize=(10, 6))
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                years = sorted(model_metrics[model_name].keys())
                values = [np.mean(model_metrics[model_name][year][metric]) if metric in model_metrics[model_name][year] and model_metrics[model_name][year][metric] else np.nan for year in years]
                plt.plot(years, values, label=metric, marker='o')
            plt.legend()
            plt.title(f'Yearwise Performance Metrics for {model_name} ({permco})')
            plt.xlabel('Year')
            plt.ylabel('Score')
            plt.xlim(min(years), max(years))
            plt.grid(True)
            plt.show()

def plot_yearwise_metrics_avg(results_dict):
    model_metrics = {model_name: {} for model_name in model_name_mapping.keys()}

    for permco, years in results_dict.items():
        for year, models in years.items():
            for model_name, metrics in models.items():
                if year not in model_metrics[model_name]:
                    model_metrics[model_name][year] = {metric: [] for metric in metrics.keys() if metric != 'confusion_matrix'}
                for metric, value in metrics.items():
                    if metric != 'confusion_matrix':
                        try:
                            model_metrics[model_name][year][metric].append(float(value))
                        except ValueError:
                            continue

    for model_name, years in model_metrics.items():
        for year, metrics in years.items():
            for metric in metrics:
                if metrics[metric]:
                    model_metrics[model_name][year][metric] = np.mean(metrics[metric])
                else:
                    model_metrics[model_name][year][metric] = np.nan

    for model_name, metrics_dict in model_metrics.items():
        years = sorted(metrics_dict.keys())
        plt.figure(figsize=(10, 6))
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [metrics_dict[year][metric] for year in years if not np.isnan(metrics_dict[year][metric])]
            valid_years = [year for year in years if not np.isnan(metrics_dict[year][metric])]
            plt.plot(valid_years, values, label=metric, marker='o')
        plt.legend()
        plt.title(f'Yearwise Performance Metrics for {model_name}')
        plt.xlabel('Year')
        plt.ylabel('Score')
        plt.xlim(min(valid_years), max(valid_years))
        plt.grid(True)
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

# Generate plots for fine-tuned model metrics
plot_yearwise_metrics_tickers(results_dict)
plot_yearwise_metrics_avg(results_dict)

year_df, overall_best_model_yearwise = find_best_model(results_dict)
plot_results_table(year_df, overall_best_model_yearwise)
plot_best_model_by_year(year_df)

ticker_df, overall_best_model_tickerwise = find_best_model_tickerwise(results_dict)
plot_results_table(ticker_df, overall_best_model_tickerwise)
plot_best_model_by_ticker(ticker_df)

plot_confusion_matrices(results_dict)