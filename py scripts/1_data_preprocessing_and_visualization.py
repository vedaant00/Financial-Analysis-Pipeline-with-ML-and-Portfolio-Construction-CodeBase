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