import os
import requests
import pandas as pd

# URL of the UCI Heart Disease (Cleveland) dataset and corresponding column names
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
        # Send HTTP GET request to download the dataset
        response = requests.get(DATA_URL)
        response.raise_for_status()
        
        # Create directories for raw and processed data if they do not exist
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        
        # Save the downloaded raw data to disk in binary format
        with open(RAW_DATA_PATH, "wb") as f:
            f.write(response.content)
        print(f"Raw data saved to {RAW_DATA_PATH}")
        
        return True
    except Exception as e:
        # Handle any download or file-related errors
        print(f"Error downloading data: {e}")
        return False

def process_data():
    """
    Loads the raw dataset, performs basic preprocessing, and
    saves the cleaned dataset for further analysis and modeling.
    """
    print("Processing data...")
    try:
        # The dataset has missing values denoted by '?'
        df = pd.read_csv(RAW_DATA_PATH, names=COLUMN_NAMES, na_values="?")
        
        print(f"Original shape: {df.shape}")
        
        # Simple preprocessing for the assignment requirements
        # 1. Drop rows with missing values (Chosen for simplicity; imputation can be applied later in the pipeline)
        df_clean = df.dropna()
        
        # 2. The target variable in Cleveland data is 0, 1, 2, 3, 4. 
        # 0 = no disease, >0 = disease. We need a binary classifier.
        df_clean.loc[:, 'target'] = df_clean['target'].apply(lambda x: 1 if x > 0 else 0)
        
        print(f"Cleaned shape: {df_clean.shape}")

        # Display class distribution after preprocessing
        print("Class distribution:")
        print(df_clean['target'].value_counts())
        
        # Save the processed dataset for EDA and model training
        df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Processed data saved to {PROCESSED_DATA_PATH}")
        
    except Exception as e:
        # Handle any errors during preprocessing
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    # Download data first; process only if download is successful
    if download_data():
        process_data()
