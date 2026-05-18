# Credit Scoring ML Labs

Exploring credit risk modelling with Kaggle datasets. The notebooks walk through end-to-end workflows for building and evaluating models that predict credit default or credit score from applicant and account features.

## Project Overview

This repo contains two related labs:

1. **Credit Score Modelling (Conor Sully dataset)**  
   Uses the `credit_score.csv` dataset from kaggle to model both:
   - `CREDIT_SCORE` as a regression target, and
   - `DEFAULT` as a classification target.

2. **German Credit Scoring Lab**  
   Uses the German Credit Data dataset to build a classification model that predicts whether a borrower is a "good" or "bad" credit risk.

Together, these labs provide hands-on practice with the credit scoring pipeline: EDA, preprocessing, model training, and evaluation.

## Notebooks

### 1. `Credit-Score.ipynb`

- Downloads the **Conor Sully credit score dataset** using `kagglehub`.
- Loads `credit_score.csv` into a Pandas DataFrame.
- Explores 80+ numerical and categorical features, including:
  - Income, savings, debt and their ratios
  - Clothing, essentials and discretionary expenditure
  - Expenditure-to-savings and expenditure-to-debt ratios
  - Categorical risk flags (e.g. gambling, credit card, mortgage, savings account, dependents)
- Experiments with:
  - Regression models for `CREDIT_SCORE`
  - Classification models for `DEFAULT`
- Compares model outputs and metrics across different approaches.

### 2. `Credit-Score-LAB-24198170.ipynb`

- Downloads the **German Credit Data** dataset from Kaggle.
- Performs initial exploration (head, shape, column names, dtypes, missing values, class distribution).
- Preprocesses features and splits the data into train/test sets.
- Trains and evaluates a **Decision Tree classifier** to predict default/good credit status.
- Uses accuracy and classification report to interpret performance.

## Datasets

1. **Conor Sully Credit Score dataset** (`credit_score.csv`)
   - Source: Kaggle – `conorsully1/credit-score`
   - Contains customer IDs, income, savings, debt, various behavioural and expenditure ratios, and targets `CREDIT_SCORE` and `DEFAULT`.

2. **German Credit Data**
   - Source: Kaggle – `varunchawla30/german-credit-data`
   - Classic credit scoring dataset with coded features and a binary credit risk label.

(See each notebook for the exact dataset links and download commands.)

## Tech Stack

- Python
- Jupyter Notebook
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- `kagglehub` / Kaggle API for dataset download

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/nizamshaikh12/credit-scoring-ml-labs.git
   cd credit-scoring-ml-labs
   ```
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter kagglehub kaggle
   ```
4. Ensure your Kaggle credentials are configured for API access (or download the datasets manually and adjust paths in the notebooks).
5. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
6. Open either `Credit-Score.ipynb` or `Credit-Score-LAB-24198170.ipynb` and run all cells in order.

## Notes

This repository focuses on **credit risk modelling** rather than pure EDA. If you want an AML EDA example instead, see the separate `synthetic-aml-transaction-monitoring-eda` repo in this portfolio.
