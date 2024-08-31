# -*- coding: utf-8 -*-
"""Comprehensive-Data-Processing-and-Machine-Learning-Pipeline-for-Financial-Analysis.py"""

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

"""## Connect to WRDS"""

import wrds
conn = wrds.Connection()

"""## CRSP Data Retrieval and Analysis

### 1. In-sample data
Historical stock data from 2005–2015 is analyzed to identify patterns and calculate financial metrics, focusing on the top 25 Consumer Discretionary companies by market capitalization.

### 2. Out-sample data
Stock data from 2016–2023 is used to validate the models developed, ensuring they perform well on new, unseen data.

"""

# Defining the date ranges for the estimation and out-of-sample periods
estimation_period_start = '2005-01-01'
estimation_period_end = '2015-12-31'
out_sample_period_start = '2016-01-01'
out_sample_period_end = '2023-12-31'

# Defining the SIC code range for the Consumer Discretionary Sector
sic_code_start = '5500'
sic_code_end = '6000'

"""## In-sample Data Retrieval and Analysis

The in-sample data consists of CRSP stock data from 2005 to 2015 for companies in the Consumer Discretionary sector. This data is retrieved and processed to identify the top 25 companies based on average market capitalization.

### Key Steps:
1. **Data Retrieval**: The CRSP dataset is queried for relevant stock data, including prices, volume, and returns, filtered by the Consumer Discretionary sector's SIC code.
2. **Data Preparation**: Dates are converted to a datetime format, and market capitalization is calculated for each company.
3. **Top 25 Companies**: The top 25 companies by average market capitalization are identified and filtered from the dataset.
4. **Data Cleaning**: The dataset is further cleaned by removing rows with missing values and duplicates.
5. **Data Analysis**: Summary statistics and checks for missing values are performed to ensure data quality.
"""

# Retrieving CRSP data for the estimation period (In-sample)
crsp_data_estimation = conn.raw_sql(f"""
    SELECT
        stock.cusip,
        stock.permno,
        stock.permco,
        company.comnam,
        company.ticker,
        stock.hsiccd,
        stock.date,
        stock.prc,
        stock.vol,
        stock.cfacpr,
        stock.shrout,
        stock.ret,
        stock.retx
    FROM
        crsp.dsf AS stock
    JOIN
        crsp.dsenames AS company
    ON
        stock.permno = company.permno
    WHERE
        company.siccd BETWEEN '{sic_code_start}' AND '{sic_code_end}'
        AND stock.date BETWEEN '{estimation_period_start}' AND '{estimation_period_end}'
""")

# Converting the 'date' column to datetime format
crsp_data_estimation['date'] = pd.to_datetime(crsp_data_estimation['date'])

# Calculating market capitalization
crsp_data_estimation['market_cap'] = crsp_data_estimation['prc'] * crsp_data_estimation['shrout']

# Calculating the average market capitalization for each company
company_avg_market_cap = crsp_data_estimation.groupby('permno').agg({'market_cap': 'mean', 'comnam': 'first'}).reset_index()

# Selecting the top 25 companies based on average market capitalization
top_25_companies = company_avg_market_cap.nlargest(25, 'market_cap')
top_25_permnos = top_25_companies['permno'].tolist()

# Filtering the original estimation dataset to include only the top 25 companies
top_25_estimation_data = crsp_data_estimation[crsp_data_estimation['permno'].isin(top_25_permnos)]

# Displaying the information about the filtered dataset
top_25_estimation_data.info()

# Dropping rows with missing values
top_25_estimation_data_clean = top_25_estimation_data.dropna()

# Removing duplicate rows based on all columns
top_25_estimation_data_clean.drop_duplicates(inplace=True)

# Displaying the information about the cleaned dataset
top_25_estimation_data_clean.info()

# Summary Statistics
summary_statistics = top_25_estimation_data_clean.describe()
print("Summary Statistics:\n", summary_statistics)

# Checking for missing values
missing_values = top_25_estimation_data_clean.isnull().sum()
print("Missing Values:\n", missing_values)

"""## Out-sample Data Retrieval and Analysis

The out-sample data, covering the period from 2016 to 2023, focuses on the same top 25 companies identified during the in-sample period. This dataset is essential for validating the models' predictive performance in a more recent and unseen data context.

### Key Steps:
1. **Data Retrieval**: The CRSP dataset is queried for the out-sample period, filtering for companies in the Consumer Discretionary sector.
2. **Data Preparation**: Dates are converted to a datetime format, and market capitalization is calculated for each company.
3. **Top 25 Companies**: The dataset is filtered to include only the top 25 companies identified during the in-sample period.
4. **Data Cleaning**: Rows with missing values are removed, and duplicates are dropped to ensure data quality.
5. **Data Analysis**: Summary statistics and checks for missing values are performed on the cleaned out-sample dataset to validate its readiness for analysis.
"""

# Retrieving CRSP data for the out-of-sample period
crsp_out_sample_data = conn.raw_sql(f"""
    SELECT
        stock.cusip,
        stock.permno,
        stock.permco,
        company.comnam,
        company.ticker,
        stock.hsiccd,
        stock.date,
        stock.prc,
        stock.vol,
        stock.cfacpr,
        stock.shrout,
        stock.ret,
        stock.retx
    FROM
        crsp.dsf AS stock
    JOIN
        crsp.dsenames AS company
    ON
        stock.permno = company.permno
    WHERE
        company.siccd BETWEEN '{sic_code_start}' AND '{sic_code_end}'
        AND stock.date BETWEEN '{out_sample_period_start}' AND '{out_sample_period_end}'
""")

# Converting the 'date' column to datetime format
crsp_out_sample_data['date'] = pd.to_datetime(crsp_out_sample_data['date'])

# Filtering the out-of-sample dataset to include only the top 25 companies identified in the estimation period
out_sample_top_25_data = crsp_out_sample_data[crsp_out_sample_data['permno'].isin(top_25_permnos)]

# Calculating market capitalization
out_sample_top_25_data['market_cap'] = out_sample_top_25_data['prc'] * out_sample_top_25_data['shrout']

# Dropping rows with missing values
out_sample_top_25_data_clean = out_sample_top_25_data.dropna()

# Removing duplicate rows based on all columns
out_sample_top_25_data_clean.drop_duplicates(inplace=True)

# Displaying the information about the cleaned out-of-sample dataset
out_sample_top_25_data_clean.info()

out_sample_summary_statistics = out_sample_top_25_data_clean.describe()
print("Out-of-Sample Data Summary Statistics:\n", out_sample_summary_statistics)

out_sample_missing_values = out_sample_top_25_data_clean.isnull().sum()
print("Missing Values after Cleaning (Out-of-Sample Data):\n", out_sample_missing_values)

"""## Sorting and Saving Cleaned Data

After cleaning the data, it's essential to ensure that the dataset is organized chronologically. This section involves sorting the estimation and out-of-sample data by date to maintain the temporal sequence, which is critical for accurate analysis and forecasting. The sorted datasets are then saved for future use.

### Key Steps:

1. **Sorting Data by Date**:
   - Both the cleaned estimation (in-sample) and out-of-sample datasets are sorted by the date column. This step ensures that all data points are in chronological order, which is crucial for time series analysis and modeling.

2. **Saving Sorted Data**:
   - The sorted datasets are saved as CSV files, making them easily accessible for further analysis and model training. This step is particularly important as it preserves the data in a structured format, ready for use in subsequent stages of the research.
"""

# Sorting the cleaned in-sample data by date
top_25_estimation_data_clean_sorted = top_25_estimation_data_clean.sort_values(by='date')

# Sorting the cleaned out-sample data by date
out_sample_top_25_data_clean_sorted = out_sample_top_25_data_clean.sort_values(by='date')

# Saving the sorted in-sample data to a CSV file for future use
top_25_estimation_data_clean_sorted.to_csv('/content/top_25_data_estimation_period.csv', index=False)

# Saving the sorted out-sample data to a CSV file for future use
out_sample_top_25_data_clean_sorted.to_csv('/content/top_25_data_outsample_period.csv', index=False)

"""## Visualization of Market Capitalization

To gain a deeper understanding of the market capitalization trends for the top 25 companies, several visualizations were created. These plots provide insights into the distribution and changes in market capitalization over time, both during the in-sample and out-sample periods.

### Key Visualizations:

1. **Distribution of Market Capitalization**:
   - A histogram was plotted to visualize the distribution of market capitalization across the top 25 companies.
   - This helps identify how market capitalization is spread among these companies, highlighting any significant variations.

2. **Average Market Capitalization Over Time**:
   - Time series plots were generated to observe the average market capitalization over the estimation and out-sample periods.
   - These plots show trends and fluctuations in market capitalization, providing a clear view of how the overall market has evolved over time.

3. **Top 25 Companies by Market Capitalization**:
   - A simple listing of the top 25 companies based on market capitalization during both the in-sample and out-sample periods.
   - This allows for a direct comparison of which companies consistently maintained high market capitalization.
"""

# Plotting the distribution of market capitalization for the top 25 companies
plt.figure(figsize=(10, 6))
plt.hist(top_25_estimation_data_clean['market_cap'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Market Capitalization for Top 25 Companies')
plt.xlabel('Market Capitalization')
plt.ylabel('Frequency')
plt.show()

# Plotting the average market capitalization over time for the in-sample period
avg_market_cap_time_series = top_25_estimation_data_clean_sorted.groupby('date')['market_cap'].mean()
plt.figure(figsize=(14, 7))
plt.plot(avg_market_cap_time_series, color='blue')
plt.title('Average Market Capitalization Over Time')
plt.xlabel('Date')
plt.ylabel('Average Market Capitalization')
plt.show()

# Plotting the average market capitalization over time for the out-sample period
avg_market_cap_time_series = out_sample_top_25_data_clean_sorted.groupby('date')['market_cap'].mean()
plt.figure(figsize=(14, 7))
plt.plot(avg_market_cap_time_series, color='blue')
plt.title('Average Market Capitalization Over Time')
plt.xlabel('Date')
plt.ylabel('Average Market Capitalization')
plt.show()

# Displaying the top 25 companies by market capitalization during the in-sample period
print("Top 25 Companies based on Market Capitalization during the In-Sample Period:")
top_25_estimation_data_clean_sorted['comnam'][:25]

# Displaying the top 25 companies by market capitalization during the out-sample period
print("\nTop 25 Companies based on Market Capitalization during the Out-Sample Period:")
out_sample_top_25_data_clean_sorted['comnam'][:25]

# Listing available tables in the 'crsp' library
crsp_tables = conn.list_tables(library='crsp')
print("Available CRSP tables:\n", crsp_tables)

"""## Extracting and Storing GV Key Data

For linking financial data across different datasets, obtaining unique identifiers like GV Keys is essential. This section outlines the process of fetching, cleaning, and storing GV Key data, which is crucial for accurate merging of datasets.

### Key Steps:

1. **Fetching GV Key Data**:
   - GV Key values are retrieved for the top 25 companies, identified by their `permnos`, from the CRSP database. GV Keys serve as unique identifiers, allowing for precise matching across different financial datasets.

2. **Cleaning and Removing Duplicates**:
   - The fetched GV Key data is checked for duplicates, and any redundant entries are removed. This ensures that each company is represented by a unique GV Key, preventing potential errors during data merging.

3. **Saving Cleaned GV Key Data**:
   - The cleaned GV Key data is saved as a CSV file for easy access in subsequent steps. This file will be used to merge the CRSP data with other datasets, ensuring accurate alignment based on GV Keys.

By securing and cleaning the GV Key data, this step ensures that the integrity of the dataset is maintained when merging with other data sources. The stored GV Key file provides a reliable reference for future data operations.
"""

# Fetching the gvkey values for the top 25 permnos from crsp.ccm_lookup
gvkey_data_query = f"""
    SELECT gvkey, lpermno, lpermco
    FROM crsp.ccm_lookup
    WHERE lpermno IN ({','.join(map(str, top_25_permnos))})
"""
gvkey_data = conn.raw_sql(gvkey_data_query)

# Displaying the fetched GV Key data
print("Fetched GV Key Data:\n", gvkey_data.head())

# Removing duplicate rows from the GV Key data
gvkey_data_unique = gvkey_data.drop_duplicates()

# Displaying the unique GV Key data
print("Unique GV Key Data:\n", gvkey_data_unique.head())

gvkeys = gvkey_data_unique['gvkey'].tolist()

# Saving the unique GV Key data to a CSV file for further use
gvkey_data_unique.to_csv('/content/gvkey_data_unique.csv', index=False)

"""## Extracting Key Developments Data from Capital IQ

The process of retrieving Key Developments data from the Capital IQ database involves several critical steps, focusing on identifying, extracting, and preparing the data for analysis. This section details how key financial events are pulled from the database for both in-sample and out-of-sample periods.

### Key Steps:

1. **Identifying Available Tables**:
   - The first step involves listing the available tables in the `ciq_keydev` library within the Capital IQ database. This ensures that the relevant data sources are identified before proceeding with data extraction.

2. **Defining Date Ranges**:
   - Specific date ranges are defined to categorize the data into in-sample (2005-2015) and out-of-sample (2016-2023) periods. These date ranges are crucial for segmenting the data for various analytical purposes.

3. **Extracting Key Development Data**:
   - Queries are executed to fetch key development data from the `ciq_keydev.wrds_keydev` table for both the estimation and out-of-sample periods. This data includes significant financial events, such as mergers, acquisitions, and earnings announcements, associated with the companies identified by their GV Keys.

4. **Retrieving Additional Event Details**:
   - A subsequent query is performed to retrieve specific details such as `keydevid`, `headline`, `situation`, and `announceddate` from the `ciq_keydev.ciqkeydev` table, based on unique `keydevids` obtained from the initial extraction.

By methodically pulling and preparing this data, the dataset becomes enriched with detailed financial events, which can then be used to conduct in-depth analysis and model development.
"""

# Listing available tables in the 'ciq_keydev' library
ciq_keydev_tables = conn.list_tables(library='ciq_keydev')

# Displaying the available Capital IQ Key Developments tables
print("Available Capital IQ Key Developments Tables:\n", ciq_keydev_tables)

# Defining the date ranges
estimation_start_date = '2005-01-01'
estimation_end_date = '2015-12-31'
out_sample_start_date = '2016-01-01'
end_date = '2023-12-31'

# Fetching key development data for the estimation period from ciq_keydev.wrds_keydev (ciqkeydev)
keydev_data_estimation_query = f"""
    SELECT *
    FROM ciq_keydev.wrds_keydev
    WHERE gvkey IN ({','.join("'" + str(gvkey) + "'" for gvkey in gvkeys)})
    AND announcedate BETWEEN '{estimation_start_date}' AND '{estimation_end_date}'
"""
keydev_data_estimation = conn.raw_sql(keydev_data_estimation_query)

# Fetching key development data for the out-of-sample period from ciq_keydev.wrds_keydev
keydev_data_out_sample_query = f"""
    SELECT *
    FROM ciq_keydev.wrds_keydev
    WHERE gvkey IN ({','.join("'" + str(gvkey) + "'" for gvkey in gvkeys)})
    AND announcedate BETWEEN '{out_sample_start_date}' AND '{end_date}'
"""
keydev_data_out_sample = conn.raw_sql(keydev_data_out_sample_query)

unique_keydevids = keydev_data_estimation["keydevid"].unique()

query = f"""
    SELECT keydevid, headline, situation, announceddate
    FROM ciq_keydev.ciqkeydev
    WHERE keydevid IN ({','.join([f"'{str(k)}'" for k in unique_keydevids])})
"""

data_sample = conn.raw_sql(query)
data_sample.head()

"""## Merging and Saving Key Development Data

After extracting the relevant Key Developments data from Capital IQ, the next critical step is merging these datasets to create comprehensive data files that include both the situational context and headlines associated with each key development event.

### Key Steps:

1. **Merging Datasets**:
   - The datasets retrieved for both the estimation and out-of-sample periods are merged with the supplementary data containing `situation` and `headline` details. This merge is based on the `keydevid` and `headline` fields, ensuring that each financial event is enriched with the corresponding descriptive information.

2. **Reviewing the Merged Dataset**:
   - After merging, a sample of the resulting dataset is displayed to verify that the merge operation was successful and that the data is correctly aligned.

3. **Saving the Merged Data**:
   - The final merged datasets for both the estimation and out-of-sample periods are saved as CSV files. These files serve as the basis for further analysis, providing a comprehensive view of the financial events and their associated details over the defined periods.
"""

# Merging the datasets based on 'keydevid'
keydev_data_estimation_merged_dataset = pd.merge(keydev_data_estimation, data_sample[['keydevid', 'situation', 'headline']],
                          on=['keydevid', 'headline'], how='left')

keydev_data_out_sample_merged_dataset = pd.merge(keydev_data_out_sample, data_sample[['keydevid', 'situation', 'headline']],
                          on=['keydevid', 'headline'], how='left')

# Display the resulting DataFrame
print("Merged Dataset Head:")
keydev_data_estimation_merged_dataset.head()

# Saving the fetched data to CSV files for further use
keydev_data_estimation_merged_dataset.to_csv('/content/keydev_data_estimation.csv', index=False)
keydev_data_out_sample_merged_dataset.to_csv('/content/keydev_data_out_sample.csv', index=False)

"""## Loading Processed Data for Analysis

Before proceeding with the analysis, the previously saved datasets need to be loaded back into the environment. These datasets include the top 25 companies' data, GV Keys, and Key Developments data for both the estimation and out-of-sample periods.

### Key Steps:

1. **Loading Top 25 Companies' Data**:
   - The data for the top 25 companies identified during the estimation and out-of-sample periods are loaded from their respective CSV files. These datasets contain detailed financial metrics that will be used for further analysis.

2. **Loading GV Key Data**:
   - The unique GV Key data, which maps the top 25 companies to their respective GV Keys, is also loaded. This data is crucial for linking financial data across different datasets.

3. **Extracting GV Keys**:
   - The GV Keys are extracted into a list to facilitate their use in further database queries or operations.

4. **Loading Key Developments Data**:
   - The Key Developments data, which details significant financial events for the top 25 companies, is loaded for both the estimation and out-of-sample periods. This data will be used to analyze the impact of these events on stock prices and market behavior.
"""

# Loading the top 25 estimation data from the CSV file
top_25_estimation_data = pd.read_csv('/content/top_25_data_estimation_period.csv')

# Loading the top 25 out-of-sample data from the CSV file
top_25_out_sample_data = pd.read_csv('/content/top_25_data_outsample_period.csv')

# Loading the unique GV Key data from the CSV file
gvkey_data_unique = pd.read_csv('/content/gvkey_data_unique.csv')

# Extracting the GV Keys into a list
gvkeys = gvkey_data_unique['gvkey'].tolist()

# Displaying the list of GV Keys
print("List of GV Keys:\n", gvkeys)

# Loading the key development data for the estimation period from the CSV file
keydev_data_estimation = pd.read_csv('/content/keydev_data_estimation.csv')

# Loading the key development data for the out-of-sample period from the CSV file
keydev_data_out_sample = pd.read_csv('/content/keydev_data_out_sample.csv')

"""## Merging Key Development Data with GV Key Information

To enhance the dataset for analysis, it's essential to merge the Key Development data with the GV Key data. This step helps in associating the financial events with specific companies by linking them through unique identifiers.

### Key Steps:

1. **Merging for the Estimation Period**:
   - The Key Development data for the estimation period is merged with the GV Key data to include `lpermno` and `lpermco` identifiers. This merge ensures that each financial event is accurately linked to the corresponding company.

2. **Merging for the Out-of-Sample Period**:
   - Similarly, the Key Development data for the out-of-sample period is merged with the GV Key data. This merge is crucial for maintaining consistency in the analysis and ensuring that the same identifiers are used across both periods.
"""

# Merging key development data with gvkey_data_unique to get lpermno and lpermco for estimation period
merged_data_estimation = pd.merge(keydev_data_estimation, gvkey_data_unique, on='gvkey', how='left')

# Merging key development data with gvkey_data_unique to get lpermno and lpermco for out-of-sample period
merged_data_out_sample = pd.merge(keydev_data_out_sample, gvkey_data_unique, on='gvkey', how='left')

print(merged_data_estimation.head(2))
print(merged_data_out_sample.head(2))

"""## Refining the Key Development Data

After merging the Key Development data with GV Key information, the next step is to refine the dataset by selecting the most relevant columns and ensuring consistency in the data formats.

### Key Steps:

1. **Column Selection**:
   - From the merged datasets, only the most pertinent columns are retained. These include identifiers (`keydevid`, `companyid`, `companyname`), event details (`headline`, `situation`, `keydeveventtypeid`, `eventtype`, `announcedate`, `announcetime`, `mostimportantdateutc`), and company-specific identifiers (`gvkey`, `lpermco`, `lpermno`).

2. **Date Conversion**:
   - The `announcedate` column in both the estimation and out-of-sample datasets is converted to a datetime format to ensure accurate time-based analysis.

3. **Column Renaming**:
   - To maintain uniformity, the columns `lpermco` and `lpermno` are renamed to `permco` and `permno` respectively in both datasets.

4. **Duplicate Removal**:
   - Any duplicate rows within the estimation and out-of-sample datasets are removed, ensuring that the data is clean and ready for subsequent analysis.
"""

# Define the selected columns to keep
selected_columns = ['keydevid', 'companyid', 'companyname', 'headline', 'situation', 'keydeveventtypeid', 'eventtype', 'announcedate', 'announcetime', 'mostimportantdateutc', 'gvkey', 'lpermco', 'lpermno']

# Select the relevant columns for the estimation period
key_data_estimation = merged_data_estimation[selected_columns]

# Select the relevant columns for the out-of-sample period
key_data_out_sample = merged_data_out_sample[selected_columns]

# Convert the 'announcedate' column to datetime format for both datasets
key_data_estimation['announcedate'] = pd.to_datetime(key_data_estimation['announcedate'])
key_data_out_sample['announcedate'] = pd.to_datetime(key_data_out_sample['announcedate'])

# Rename columns 'lpermco' and 'lpermno' to 'permco' and 'permno' for the estimation period
key_data_estimation.rename(columns={'lpermco': 'permco', 'lpermno': 'permno'}, inplace=True)

# Rename columns 'lpermco' and 'lpermno' to 'permco' and 'permno' for the out-of-sample period
key_data_out_sample.rename(columns={'lpermco': 'permco', 'lpermno': 'permno'}, inplace=True)

# Drop duplicate rows in the estimation dataset
key_data_estimation.drop_duplicates(inplace=True)

# Drop duplicate rows in the out-of-sample dataset
key_data_out_sample.drop_duplicates(inplace=True)

# Display information about the cleaned estimation dataset
print(key_data_estimation.info())

# Display information about the cleaned out-of-sample dataset
print(key_data_out_sample.info())

"""## Event Type Analysis in Key Developments

This section highlights the most common financial events within the Key Developments dataset.

### Key Steps:

1. **Top 20 Event Types**:
   - The most frequent event types are identified and selected for analysis.

2. **Visualization**:
   - A bar plot visualizes the distribution of these top 20 event types, using a clear and engaging color scheme.
"""

# Calculating the top 20 most frequent event types in the estimation dataset
event_counts = key_data_estimation['eventtype'].value_counts().nlargest(20).reset_index()
event_counts.columns = ['eventtype', 'count']

# Plotting the distribution of the top 20 event types using Matplotlib
plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='eventtype', data=event_counts, palette='viridis')

# Adding title and labels
plt.title('Top 20 Distribution of Event Types in Capital IQ Key Developments')
plt.xlabel('Event Count')
plt.ylabel('Event Type')

# Displaying the plot
plt.show()

"""## Preparing In-Sample Data for Weekly Analysis

In this section, we focus on filtering and preparing the weekly CRSP data for the in-sample period to facilitate further analysis.

### Key Steps

1. **Data Filtering**:
   - Filtered out entries with null or negative prices and returns to ensure the quality and accuracy of the dataset.

2. **Adjusted Price Calculation**:
   - Calculated adjusted prices by dividing the raw price by the cumulative adjustment factor (`cfacpr`). This adjustment is crucial to account for stock splits, dividends, and other corporate actions.

3. **Weekly Aggregation**:
   - Grouped the data by `permco`, `ticker`, and weekly frequency (with the week ending on Friday). The last adjusted price of the week was selected, and weekly returns were calculated using the log-return formula, ensuring the data reflects weekly market movements accurately.

4. **Data Structuring**:
   - Renamed columns for better clarity and selected only the relevant columns (`permco`, `ticker`, `start_date`, `end_date`, `weekly_ret`, `adj_prc`) to focus on the necessary data for subsequent analysis.
"""

# Filter out null and negative prices and returns
crsp_train_filtered = top_25_estimation_data[(top_25_estimation_data['prc'] > 0) & ~top_25_estimation_data['ret'].isna()]

# Calculate adjusted price
crsp_train_filtered['adj_prc'] = crsp_train_filtered['prc'] / crsp_train_filtered['cfacpr']

# Set date column to datetime type
crsp_train_filtered['date'] = pd.to_datetime(crsp_train_filtered['date'])

# Group by permco, ticker, and weekly frequency ending on Friday
weekly_data = crsp_train_filtered.groupby(['permco', 'ticker', pd.Grouper(key='date', freq='W-FRI')]).agg({
    'adj_prc': lambda x: x.iloc[-1],  # Adjusted price at the end of the week
    'ret': (lambda x: (np.exp(np.sum(np.log(1 + x))) - 1) * 100)  # Weekly return calculation
}).reset_index()

# Rename columns for clarity
weekly_data.rename(columns={'date': 'end_date', 'ret': 'weekly_ret'}, inplace=True)

# Add start_date column (beginning of the week)
weekly_data['start_date'] = weekly_data['end_date'] - pd.DateOffset(days=4)

# Select relevant columns
weekly_data_crsp_insample = weekly_data[['permco', 'ticker', 'start_date', 'end_date', 'weekly_ret', 'adj_prc']]

# Display the resulting DataFrame
weekly_data_crsp_insample.head()

"""## Calculating Price Direction for In-Sample Data

This section focuses on determining the weekly price movement direction based on adjusted prices within the in-sample dataset.

### Key Steps

1. **Direction Calculation**:
   - Calculated the price direction by computing the difference in adjusted prices (`adj_prc`) for each `permco` and `ticker`. If the difference was negative, the direction was marked as 'Down', otherwise 'Up'. This classification helps in understanding the weekly price trends.

2. **DataFrame Structuring**:
   - The final DataFrame, `result_df`, includes the relevant columns: `permco`, `ticker`, `start_date`, `end_date`, `weekly_ret`, `adj_prc`, and the calculated `direction`, which will be used for further analysis.
"""

# Calculate direction based on adjusted prices
weekly_data_crsp_insample['direction'] = weekly_data_crsp_insample.groupby(['permco', 'ticker'])['adj_prc'].diff().apply(lambda x: 'Down' if x < 0 else 'Up')

# Display the resulting DataFrame with direction
result_df = weekly_data_crsp_insample[['permco', 'ticker', 'start_date', 'end_date', 'weekly_ret', 'adj_prc', 'direction']]
result_df.head()

"""## Preparing Out-Sample Data for Weekly Analysis

This section outlines the steps taken to clean and prepare the out-sample data for analysis of weekly returns and adjusted prices.

### Key Steps

1. **Data Filtering**:
   - Filtered out records with null or negative prices (`prc`) and returns (`ret`). This ensures that only valid and meaningful data is used for analysis.

2. **Adjusted Price Calculation**:
   - Calculated the adjusted price (`adj_prc`) by dividing the original price (`prc`) by the price adjustment factor (`cfacpr`). This adjustment accounts for events like stock splits and dividends, providing a more accurate measure of price.

3. **Weekly Aggregation**:
   - Grouped the data by `permco`, `ticker`, and weekly frequency ending on Friday. The last adjusted price of the week was selected, and the weekly return was calculated using a log-based approach to capture the compounded return over the week.

4. **Data Structuring**:
   - Renamed and added relevant columns, including `start_date` (beginning of the week) and `end_date` (end of the week). The final DataFrame, `weekly_data_crsp_outsample`, includes columns such as `permco`, `ticker`, `start_date`, `end_date`, `weekly_ret`, and `adj_prc`, setting the stage for further analysis.
"""

# Filter out null and negative prices and returns
crsp_train_filtered_out = top_25_out_sample_data[(top_25_out_sample_data['prc'] > 0) & ~top_25_out_sample_data['ret'].isna()]

# Calculate adjusted price
crsp_train_filtered_out['adj_prc'] = crsp_train_filtered_out['prc'] / crsp_train_filtered_out['cfacpr']

# Set date column to datetime type
crsp_train_filtered_out['date'] = pd.to_datetime(crsp_train_filtered_out['date'])

# Group by permco, ticker, and weekly frequency ending on Friday
weekly_data_out = crsp_train_filtered_out.groupby(['permco', 'ticker', pd.Grouper(key='date', freq='W-FRI')]).agg({
    'adj_prc': lambda x: x.iloc[-1],  # Adjusted price at the end of the week
    'ret': (lambda x: (np.exp(np.sum(np.log(1 + x))) - 1) * 100)  # Weekly return calculation
}).reset_index()

# Rename columns for clarity
weekly_data_out.rename(columns={'date': 'end_date', 'ret': 'weekly_ret'}, inplace=True)

# Add start_date column (beginning of the week)
weekly_data_out['start_date'] = weekly_data_out['end_date'] - pd.DateOffset(days=4)

# Select relevant columns
weekly_data_crsp_outsample = weekly_data_out[['permco', 'ticker', 'start_date', 'end_date', 'weekly_ret', 'adj_prc']]

# Display the resulting DataFrame
weekly_data_crsp_outsample.head()

"""## Directional Analysis of Out-Sample Data

This section describes the process used to calculate the direction of stock price movements based on adjusted prices and to prepare the final dataset for analysis.

### Key Steps

1. **Direction Calculation**:
   - Calculated the direction of price movement by assessing the difference in adjusted prices (`adj_prc`) for each stock (`permco`, `ticker`). If the price decreased compared to the previous period, it was labeled as 'Down'; otherwise, it was labeled as 'Up'.

2. **DataFrame Structuring**:
   - Created a DataFrame, `result_outsample_df`, which includes the key variables such as `permco`, `ticker`, `start_date`, `end_date`, `weekly_ret`, `adj_prc`, and the calculated `direction`. This structured dataset provides a comprehensive view of the weekly returns and the direction of price movements, ready for further analysis.
"""

# Calculate direction based on adjusted prices
weekly_data_crsp_outsample['direction'] = weekly_data_crsp_outsample.groupby(['permco', 'ticker'])['adj_prc'].diff().apply(lambda x: 'Down' if x < 0 else 'Up')

# Display the resulting DataFrame with direction
result_outsample_df = weekly_data_crsp_outsample[['permco', 'ticker', 'start_date', 'end_date', 'weekly_ret', 'adj_prc', 'direction']]
result_outsample_df.head()

"""## Visualizing Weekly Returns by Ticker (In-Sample Data)

This section illustrates the steps taken to visualize the weekly returns for each ticker in the in-sample data. The process involves grouping the data by ticker and creating subplots to display the weekly returns over time.

### Key Steps

1. **Grouping and Preparation**:
   - Converted the `start_date` column to datetime format to ensure proper time series plotting.
   - Grouped the data by `ticker` to prepare for individual plots for each company's stock.

2. **Subplot Configuration**:
   - Calculated the number of rows and columns needed for the subplots, depending on the number of tickers.
   - Created a subplot grid, adjusting the layout to fit all tickers while ensuring clarity and readability.

3. **Plotting**:
   - Assigned unique colors to each ticker for visual distinction.
   - Plotted the weekly returns (`weekly_ret`) against `start_date` for each ticker.
   - Adjusted the subplot layout, hiding any unused plots, and added titles and labels to ensure the plot is informative.

4. **Final Visualization**:
   - The final plot, titled "Weekly Returns by Ticker," offers a comprehensive view of the performance of each ticker over time within the in-sample period.
"""

weekly_data_crsp_insample['start_date'] = pd.to_datetime(weekly_data_crsp_insample['start_date'])

# Group the data by ticker
grouped_data = weekly_data_crsp_insample.groupby('ticker')

# Calculate the number of rows and columns for subplots
num_tickers = len(grouped_data)
cols = 2
rows = (num_tickers + cols - 1) // cols

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), sharex=True)
axes = axes.flatten()

# Generate a list of unique colors
colors = np.random.rand(num_tickers, 3)

# Iterate over each ticker group and assign a unique color
for idx, (ticker, data) in enumerate(grouped_data):
    ax = axes[idx]
    ax.plot(data['start_date'], data['weekly_ret'], color=colors[idx], label=ticker)
    ax.set_title(ticker)
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Return')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend()

# Hide any unused subplots
for i in range(num_tickers, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout
fig.tight_layout()
fig.suptitle('Weekly Returns by Ticker', fontsize=16)
fig.subplots_adjust(top=0.95)  # Adjust top to fit the title

# Show the plot
plt.show()

"""## Visualizing Weekly Returns by Ticker (Out-of-Sample Data)

This section details the steps used to visualize the weekly returns for each ticker in the out-of-sample data. The process involves organizing the data by ticker and creating subplots for a comprehensive visualization of each stock's weekly performance.

### Key Steps

1. **Data Preparation**:
   - The `start_date` column was converted to datetime format to facilitate accurate time series plotting.
   - The data was grouped by `ticker` to allow for individual plots for each company's stock returns.

2. **Subplot Configuration**:
   - Calculated the appropriate number of rows and columns required to accommodate the number of tickers in the dataset.
   - A grid of subplots was created, with adjustments made to the layout to ensure clarity and effective use of space.

3. **Plotting**:
   - Each ticker was assigned a unique color to distinguish it visually in the plots.
   - The weekly returns (`weekly_ret`) were plotted against `start_date` for each ticker, providing a detailed view of stock performance over time.
   - The subplot layout was fine-tuned by hiding any unused plots and adding relevant titles and labels.

4. **Final Visualization**:
   - The final visualization, titled "Weekly Returns by Ticker (Out-of-Sample)," presents a clear and organized view of each ticker's performance throughout the out-of-sample period.
"""

weekly_data_crsp_outsample['start_date'] = pd.to_datetime(weekly_data_crsp_outsample['start_date'])

# Group the data by ticker
grouped_data = weekly_data_crsp_outsample.groupby('ticker')

# Calculate the number of rows and columns for subplots
num_tickers = len(grouped_data)
cols = 2
rows = (num_tickers + cols - 1) // cols

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), sharex=True)
axes = axes.flatten()

# Generate a list of unique colors
colors = np.random.rand(num_tickers, 3)

# Iterate over each ticker group and assign a unique color
for idx, (ticker, data) in enumerate(grouped_data):
    ax = axes[idx]
    ax.plot(data['start_date'], data['weekly_ret'], color=colors[idx], label=ticker)
    ax.set_title(ticker)
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Return')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend()

# Hide any unused subplots
for i in range(num_tickers, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout
fig.tight_layout()
fig.suptitle('Weekly Returns by Ticker (Out-of-Sample)', fontsize=16)
fig.subplots_adjust(top=0.95)  # Adjust top to fit the title

# Show the plot
plt.show()

"""## Merging Key Development Data with CRSP Weekly Data (In-Sample)

This section describes the process of preparing and merging key development data with CRSP weekly data for in-sample analysis. The goal is to align financial events with stock performance to enhance the predictive model's inputs.

### Key Steps

1. **Preparing the Key Development Data**:
   - The `announcedate` column was converted to datetime format for accurate time-based operations.
   - Each announcement date was mapped to the corresponding Friday of the week (`fri_of_week`).
   - A new column, `next_fri_of_week`, was created to represent the Friday of the following week, providing a future reference point for event analysis.

2. **Preparing the CRSP Weekly Data**:
   - The `end_date` column in the CRSP dataset was also converted to datetime format.
   - Each `end_date` was aligned with the corresponding Friday of the week (`fri_of_week`) to facilitate the merging process.

3. **Merging the Datasets**:
   - The Key Development Data was merged with the CRSP Weekly Data using `permco` and `fri_of_week` as common keys.
   - The merge operation was performed as a left join, ensuring that all CRSP weekly records were retained even if no corresponding key development data was found.

4. **Handling Missing Data**:
   - Missing headlines in the merged dataset were filled with the placeholder 'No_Headlines' to maintain consistency.

5. **Final Structuring**:
   - The merged dataset was refined by selecting relevant columns, including `permco`, `ticker`, `start_date`, `end_date`, `weekly_ret`, `adj_prc`, `headline`, and `direction`.
   - The resulting DataFrame, `in_sample_structured`, was then displayed for validation.
"""

# Step 1: Prepare the Key Development Data
key_data_estimation['announcedate'] = pd.to_datetime(key_data_estimation['announcedate'])
key_data_estimation['fri_of_week'] = key_data_estimation['announcedate'].dt.to_period('W-FRI').apply(lambda r: r.end_time).dt.date
key_data_estimation['next_fri_of_week'] = key_data_estimation['fri_of_week'] + timedelta(days=7)
key_data_estimation['next_fri_of_week'] = key_data_estimation['next_fri_of_week'].astype(str)

# Step 2: Prepare the CRSP Weekly Data
weekly_data_crsp_insample['end_date'] = pd.to_datetime(weekly_data_crsp_insample['end_date'])
weekly_data_crsp_insample['fri_of_week'] = weekly_data_crsp_insample['end_date'].dt.to_period('W-FRI').apply(lambda r: r.end_time).dt.date.astype(str)

# Step 3: Merge the Key Development Data with the CRSP Weekly Data
merged_data = pd.merge(weekly_data_crsp_insample, key_data_estimation,
                       left_on=['permco', 'fri_of_week'],
                       right_on=['permco', 'next_fri_of_week'],
                       how='left')

# Step 4: Fill Missing Headlines with 'No_Headlines'
merged_data['headline'] = merged_data['headline'].fillna('No_Headlines')

# Step 5: Select and reorder columns
in_sample_structured = merged_data[['permco', 'ticker', 'start_date', 'end_date', 'weekly_ret', 'adj_prc', 'headline', 'direction']]

# Display the resulting DataFrame
in_sample_structured.head()

"""## Merging Key Development Data with CRSP Weekly Data (Out-of-Sample)

This section outlines the steps to merge key development data with CRSP weekly data for the out-of-sample period. This integration aims to link significant financial events to stock performance data for enhanced analysis and modeling.

### Key Steps

1. **Preparing the Key Development Data**:
   - The `announcedate` column was converted to datetime format to ensure accurate temporal alignment.
   - The corresponding Friday of each announcement date was identified and stored in the `fri_of_week` column.
   - A new column, `next_fri_of_week`, was created to represent the following Friday, providing a forward-looking time reference for analyzing event impacts.

2. **Preparing the CRSP Weekly Data**:
   - The `end_date` in the CRSP dataset was also converted to datetime format for consistency.
   - Each `end_date` was aligned with the appropriate Friday (`fri_of_week`) to facilitate the merge.

3. **Merging the Datasets**:
   - The Key Development Data was merged with the CRSP Weekly Data using `permco` and `fri_of_week` as join keys.
   - A left join was employed to retain all CRSP weekly records, even if no corresponding key development data was available.

4. **Handling Missing Data**:
   - Missing headlines in the merged dataset were filled with 'No_Headlines' to ensure completeness and avoid gaps in the data.

5. **Final Structuring**:
   - The merged dataset was refined by selecting key columns, including `permco`, `ticker`, `start_date`, `end_date`, `weekly_ret`, `adj_prc`, `headline`, and `direction`.
   - The resulting DataFrame, `out_sample_structured`, was displayed for review and further analysis.
"""

# Step 1: Prepare the Key Development Data
key_data_out_sample['announcedate'] = pd.to_datetime(key_data_out_sample['announcedate'])
key_data_out_sample['fri_of_week'] = key_data_out_sample['announcedate'].dt.to_period('W-FRI').apply(lambda r: r.end_time).dt.date
key_data_out_sample['next_fri_of_week'] = key_data_out_sample['fri_of_week'] + timedelta(days=7)
key_data_out_sample['next_fri_of_week'] = key_data_out_sample['next_fri_of_week'].astype(str)

# Step 2: Prepare the CRSP Weekly Data
weekly_data_crsp_outsample['end_date'] = pd.to_datetime(weekly_data_crsp_outsample['end_date'])
weekly_data_crsp_outsample['fri_of_week'] = weekly_data_crsp_outsample['end_date'].dt.to_period('W-FRI').apply(lambda r: r.end_time).dt.date.astype(str)

# Step 3: Merge the Key Development Data with the CRSP Weekly Data
merged_data = pd.merge(weekly_data_crsp_outsample, key_data_out_sample,
                       left_on=['permco', 'fri_of_week'],
                       right_on=['permco', 'next_fri_of_week'],
                       how='left')

# Step 4: Fill Missing Headlines with 'No_Headlines'
merged_data['headline'] = merged_data['headline'].fillna('No_Headlines')

# Step 5: Select and reorder columns
out_sample_structured = merged_data[['permco', 'ticker', 'start_date', 'end_date', 'weekly_ret', 'adj_prc', 'headline', 'direction']]

# Display the resulting DataFrame
out_sample_structured.head()

"""## Tokenization of Headlines for the Data

This section outlines the steps involved in tokenizing and analyzing the textual data from stock-related headlines. By transforming the text into token counts, the analysis can identify the most common words across different stocks and in the dataset as a whole.

### Key Steps

1. **Exclusion of 'No_Headlines'**:
   - The dataset was filtered to remove entries with 'No_Headlines', ensuring that only relevant text data is analyzed.

2. **List of Stopwords**:
   - A list of common English stopwords was defined using the NLTK library. These words (e.g., 'the', 'and', 'is') are excluded from the analysis to focus on more meaningful terms.

3. **Tokenization Function**:
   - A function `tokenize_text` was defined to convert the headlines into a format suitable for analysis. The function uses `CountVectorizer` from `sklearn` to transform the text into token counts, which are then aggregated to identify the most frequent words.

4. **Tokenization by Ticker**:
   - The headlines were tokenized separately for each stock (`ticker`). This allows for a detailed analysis of the most common terms associated with each specific stock.

5. **Tokenization for All Stocks Together**:
   - Additionally, all the headlines were tokenized collectively to provide an overview of the most frequent terms across the entire dataset.

The resulting token counts offer valuable insights into the key themes and topics mentioned in the headlines, which can be further explored in subsequent analysis.

"""

# Remove 'No_Headlines' from analysis
in_sample_structured_NoHeadline = in_sample_structured[in_sample_structured['headline'] != 'No_Headlines']

# List of stopwords
stop_words = stopwords.words('english')

def tokenize_text(headlines):
    """
    Tokenizes the text in the headlines and returns the sum of token counts.
    """
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(headlines)
    token_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return token_counts.sum().sort_values(ascending=False)

# Tokenize for each stock and all together
token_counts_per_stock = {}
for ticker in in_sample_structured_NoHeadline['ticker'].unique():
    headlines = in_sample_structured_NoHeadline[in_sample_structured_NoHeadline['ticker'] == ticker]['headline']
    token_counts_per_stock[ticker] = tokenize_text(headlines)

# Tokenize for all stocks together
all_headlines = in_sample_structured_NoHeadline['headline']
total_token_counts = tokenize_text(all_headlines)

"""## Visualization of Word Clouds for Stock Headlines

This section demonstrates the process of generating word clouds, which visually represent the frequency of words in stock-related headlines. Word clouds help identify the most prominent terms associated with each stock and across the entire dataset.

### Key Steps

1. **Word Cloud Function**:
   - The `plot_word_cloud` function was defined to create and display word clouds. It takes the text data and generates a word cloud, with more frequently occurring words displayed in larger fonts. The function also removes common stopwords to ensure the focus is on significant terms.

2. **Subplot Layout Calculation**:
   - The optimal layout for the subplots was calculated based on the number of word clouds to be generated. Each subplot corresponds to a specific stock or the entire dataset.

3. **Figure Size Adjustment**:
   - The figure's dimensions were adjusted to accommodate the number of subplots. The width and height were set to ensure that the word clouds are clearly visible and well-spaced.

4. **Generating Word Clouds for Each Stock**:
   - Word clouds were generated individually for each stock, allowing for a detailed visualization of the key terms mentioned in the headlines of that specific stock.

5. **Generating a Word Cloud for All Headlines**:
   - An additional word cloud was created using all the headlines in the dataset. This provides an overall view of the most frequently mentioned words across all stocks.

6. **Hiding Unused Subplots**:
   - Any extra subplot areas that were not used were hidden to maintain a clean and focused visual presentation.

The resulting word clouds offer an insightful visual summary of the most discussed topics in the headlines, both for individual stocks and across the entire dataset.
"""

# Function to generate and plot word cloud with specified size
def plot_word_cloud(ax, text, title):
    wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)

# Calculate optimal subplot layout
num_plots = len(token_counts_per_stock) + 1  # Number of stocks + all headlines
cols = 3  # Number of columns in the layout
rows = (num_plots - 1) // cols + 1  # Calculate number of rows needed

# Adjust figure size based on the number of subplots
fig_width = 15  # Width of the figure
fig_height = rows * 5  # Height of the figure, adjusted based on the number of rows

fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

# Generate word clouds for each stock
for i, ticker in enumerate(token_counts_per_stock.keys()):
    row = i // cols
    col = i % cols
    text = ' '.join(in_sample_structured_NoHeadline[in_sample_structured_NoHeadline['ticker'] == ticker]['headline'])
    plot_word_cloud(axs[row, col], text, f'Word Cloud for {ticker}')

# Generate word cloud for all headlines together
all_text = ' '.join(in_sample_structured_NoHeadline['headline'])
plot_word_cloud(axs.flatten()[num_plots-1], all_text, 'Word Cloud for All Headlines')

# Hide any extra subplots
for i in range(num_plots, rows * cols):
    axs.flatten()[i].axis('off')

plt.tight_layout()
plt.show()

"""## Displaying Top Tokens in Stock Headlines

This section outlines the steps taken to identify and display the most frequently occurring words (tokens) in the headlines associated with each stock and across the entire dataset.

### Key Steps

1. **Top Tokens Display Function**:
   - The `display_top_tokens` function was defined to print out the most frequent tokens (words) in the headlines. This function takes in the token counts and a title for context, then outputs the top `n` tokens, where `n` is set to 10 by default.

2. **Displaying Top Tokens for Each Stock**:
   - The function was applied to the token counts of each stock. This allowed for the identification of the top 10 most frequent words in the headlines for each specific stock, providing insight into the key terms discussed for each company.

3. **Displaying Top Tokens for All Headlines**:
   - The function was also used to display the top tokens across all headlines in the dataset. This broader analysis highlights the most common themes or topics mentioned in the news across all companies.

The identification and display of these top tokens provide a concise summary of the most significant words in the dataset, helping to understand the focus of news coverage for each stock and overall.
"""

# Function to display top tokens
def display_top_tokens(token_counts, title, n=10):
    print(f"Top {n} tokens for {title}:")
    print(token_counts.head(n))
    print()

# Display top tokens for each stock
for ticker, counts in token_counts_per_stock.items():
    display_top_tokens(counts, ticker)

# Display top tokens for all headlines together
display_top_tokens(total_token_counts, 'All Headlines')

"""## Data Validation, Preprocessing, and Column Renaming

This section details the steps involved in preparing and enhancing the in-sample and out-sample datasets for further analysis.

### Key Steps

1. **Loading the Datasets**:
   - The `in_sample_structured` and `out_sample_structured` datasets were loaded into `in_sample_data` and `out_sample_data` respectively.

2. **Renaming Columns**:
   - A dictionary (`columns_rename_mapping`) was created to map the original column names to more descriptive names.
   - The `rename` function was applied to both datasets to update the column names accordingly.

3. **Ensuring Date Columns are in Datetime Format**:
   - The `Start_Date` and `End_Date` columns in both datasets were converted to datetime format to ensure accurate date-based operations.

4. **Checking for Missing Values and Duplicates**:
   - Assertions were used to ensure that no missing values were present in either dataset.
   - Duplicate rows, if any, were removed from the datasets.

5. **Mapping the 'Direction' Column to Numerical Values**:
   - The `Direction` column, originally containing values 'Up' or 'Down', was mapped to numerical values (1 for 'Up' and 0 for 'Down') for ease of analysis.

6. **Generating Additional Features**:
   - Additional features were generated as needed, such as the 5-period moving average of the adjusted price (`Adjusted_Price_MA5`).

7. **Saving the Cleaned and Enhanced Datasets**:
   - The cleaned and processed datasets were saved as CSV files for further use in analysis.

These steps ensured that the datasets were well-prepared, with consistent formatting and no missing or duplicate values, ready for subsequent modeling and analysis.
"""

# Load the in-sample and out-sample datasets
in_sample_data = in_sample_structured
out_sample_data = out_sample_structured

# Rename columns
columns_rename_mapping = {
    'index': 'Index',
    'permco': 'Permco',
    'ticker': 'Ticker',
    'start_date': 'Start_Date',
    'end_date': 'End_Date',
    'weekly_ret': 'Weekly_Return',
    'adj_prc': 'Adjusted_Price',
    'headline': 'Headline',
    'direction': 'Direction'
}

in_sample_data.rename(columns=columns_rename_mapping, inplace=True)
out_sample_data.rename(columns=columns_rename_mapping, inplace=True)

# Ensure date columns are in datetime format
in_sample_data['Start_Date'] = pd.to_datetime(in_sample_data['Start_Date'])
in_sample_data['End_Date'] = pd.to_datetime(in_sample_data['End_Date'])
out_sample_data['Start_Date'] = pd.to_datetime(out_sample_data['Start_Date'])
out_sample_data['End_Date'] = pd.to_datetime(out_sample_data['End_Date'])

# Check for missing values
assert in_sample_data.isnull().sum().sum() == 0, "In-sample data contains missing values"
assert out_sample_data.isnull().sum().sum() == 0, "Out-sample data contains missing values"

# Check for duplicate rows and remove them
in_sample_data = in_sample_data.drop_duplicates()
out_sample_data = out_sample_data.drop_duplicates()

# Convert 'Direction' to numerical values (e.g., 'Up' -> 1, 'Down' -> 0)
direction_mapping = {'Up': 1, 'Down': 0}
in_sample_data['Direction'] = in_sample_data['Direction'].map(direction_mapping)
out_sample_data['Direction'] = out_sample_data['Direction'].map(direction_mapping)

# Generate additional features if needed (e.g., moving averages, volatility)
# Example: 5-period moving average of adjusted price
in_sample_data['Adjusted_Price_MA5'] = in_sample_data['Adjusted_Price'].rolling(window=5).mean()
out_sample_data['Adjusted_Price_MA5'] = out_sample_data['Adjusted_Price'].rolling(window=5).mean()

# Save the cleaned and enhanced datasets
in_sample_data.to_csv('/content/datasets/cleaned_in_sample_data.csv', index=False)
out_sample_data.to_csv('/content/datasets/cleaned_out_sample_data.csv', index=False)

print("Data validation, preprocessing, and column renaming complete.")

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

"""## End Notes

This analysis highlights the effectiveness of both pre-trained and fine-tuned language models in predicting market movements and constructing portfolios. By calculating and comparing key metrics such as returns, volatility, and Sharpe ratios across various scenarios (with and without transaction costs), we have gained a deeper understanding of each model's performance.

The results demonstrate that while pre-trained models offer a solid baseline, fine-tuned models generally yield superior risk-adjusted returns, particularly in more nuanced market conditions. This reinforces the value of domain-specific fine-tuning in enhancing predictive accuracy and optimizing portfolio performance.
"""