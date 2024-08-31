# Financial-Analysis-Pipeline-with-ML-and-Portfolio-Construction-CodeBase
This repository contains Python scripts and Jupyter Notebooks for financial analysis, integrating data retrieval, machine learning models, and portfolio construction. It includes text embedding using transformers, rolling-window analysis, and visualisation of portfolio performance. Ideal for finance and ML research.

# 🚀 Comprehensive Data Processing and Machine Learning Pipeline for Financial Analysis 📊

Welcome to the **Comprehensive Data Processing and Machine Learning Pipeline for Financial Analysis** repository! This project provides a robust and versatile pipeline designed for financial data analysis, featuring tools for data processing, feature engineering, machine learning, and performance evaluation. The codebase is adaptable to local machines and cloud environments like Google Colab.

## 📂 Repository Contents

- **`comprehensive_data_processing_and_machine_learning_pipeline_for_financial_analysis.py`**: The main Python script for the complete pipeline.
- **`Comprehensive_Data_Processing_and_Machine_Learning_Pipeline_for_Financial_Analysis.ipynb`**: Jupyter Notebook version optimized for Google Colab.

## ✨ Features

- 🛠️ **Data Retrieval & Processing**: Seamlessly fetch and process financial data.
- 🎛️ **Feature Engineering & Model Training**: Utilize advanced techniques to prepare your data and train state-of-the-art models.
- 📈 **Performance Evaluation**: Comprehensive metrics and visualizations to analyze model performance.
- 🌐 **Local & Cloud Compatibility**: You can easily run the pipeline on your local machine or in the cloud using Google Colab.

## 🔧 Requirements

- 🐍 **Python**: 3.10.12 or above
- 📦 Required Python packages:
  - `numpy` (latest)
  - `pandas` (latest)
  - `scipy` (latest)
  - `matplotlib` (latest)
  - `seaborn` (latest)
  - `torch` (latest)
  - `transformers` (latest)
  - `datasets` (latest)
  - `wrds` (latest)
  - `accelerate` (latest)

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

## 2️⃣ Running the `.py` File Locally

### Install Dependencies 📥:

Ensure you have Python 3.10.12 or above. Install the necessary packages:

```python
pip install wrds
pip install --upgrade numpy scipy pyarrow pandas datasets transformers
pip install accelerate -U
```

### Run the Python Script 🖥️:
Execute the main script with:

```bash
python comprehensive_data_processing_and_machine_learning_pipeline_for_financial_analysis.py
```

## 3️⃣ Running the .ipynb File Locally

### Install Jupyter Notebook 📓:
If you don’t have Jupyter Notebook installed:

```bash
pip install notebook
```

### Launch Jupyter Notebook 🚀:
Start Jupyter Notebook:

```bash
jupyter notebook
```

### Open the Notebook 📂:
Navigate to the `Comprehensive_Data_Processing_and_Machine_Learning_Pipeline_for_Financial_Analysis.ipynb` file and open it.

---

## 4️⃣ Running on Google Colab

### Open Google Colab 🌐:
Head over to [Google Colab](https://colab.research.google.com/).

### Upload the Notebook 📤:
- Click on `File > Upload Notebook`.
- Upload `Comprehensive_Data_Processing_and_Machine_Learning_Pipeline_for_Financial_Analysis.ipynb`.

### Install Required Libraries ⚙️:
In the first cell, install the necessary packages:

```python
!pip install wrds
!pip install --upgrade numpy scipy pyarrow pandas datasets transformers
!pip install accelerate -U
```

### Run the Notebook ▶️:
Execute each cell to run the entire pipeline.

---

## 🛠️ Hardware Configuration

The project has been tested on the following hardware configuration:

- **GPU**: NVIDIA L4
- **CUDA Version**: 12.2
- **RAM**: 53.0 GB
- **GPU RAM**: 22.5 GB
- **Disk**: 78.2 GB

---

## 📑 Documentation

Detailed documentation for the project can be found in the `docs` folder. This includes descriptions of each function, class, and workflow within the project.
