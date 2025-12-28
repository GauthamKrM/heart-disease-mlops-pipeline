import os
import requests
import pandas as pd

# URL for the Heart Disease dataset (Cleveland database often used)
# Using a stable source. The UCI repository sometimes changes URLs.
# We will use the version hosted on a reliable raw git or standard UCI repo mirror if possible.
# For this example, we will treat the processed cleveland data.
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

RAW_DATA_PATH = "data/raw/heart_disease.csv"
PROCESSED_DATA_PATH = "data/processed/heart_disease_cleaned.csv"

def download_data():
    print(f"Downloading data from {DATA_URL}...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        
        # Save raw data
        with open(RAW_DATA_PATH, "wb") as f:
            f.write(response.content)
        print(f"Raw data saved to {RAW_DATA_PATH}")
        
        return True
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False

def process_data():
    print("Processing data...")
    try:
        # The dataset has missing values denoted by '?'
        df = pd.read_csv(RAW_DATA_PATH, names=COLUMN_NAMES, na_values="?")
        
        print(f"Original shape: {df.shape}")
        
        # Simple preprocessing for the assignment requirements
        # 1. Drop rows with missing values (or impute - we'll drop for simplicity in this script, 
        #    but the training pipeline can handle imputation if we want to be more robust. 
        #    Let's drop here to have a 'clean' baseline dataset for EDA)
        df_clean = df.dropna()
        
        # 2. The target variable in Cleveland data is 0, 1, 2, 3, 4. 
        #    0 = no disease, >0 = disease. We need a binary classifier.
        df_clean.loc[:, 'target'] = df_clean['target'].apply(lambda x: 1 if x > 0 else 0)
        
        print(f"Cleaned shape: {df_clean.shape}")
        print("Class distribution:")
        print(df_clean['target'].value_counts())
        
        df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Processed data saved to {PROCESSED_DATA_PATH}")
        
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    if download_data():
        process_data()
