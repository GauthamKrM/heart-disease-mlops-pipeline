import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@pytest.fixture
def sample_data():
    """Returns a small DataFrame with heart disease columns for testing."""
    data = {
        "age": [63, 37, 41, 56, 57],
        "sex": [1, 1, 0, 1, 0],
        "cp": [3, 2, 1, 1, 0],
        "trestbps": [145, 130, 130, 120, 120],
        "chol": [233, 250, 204, 236, 354],
        "fbs": [1, 0, 0, 0, 0],
        "restecg": [0, 1, 0, 1, 1],
        "thalach": [150, 187, 172, 178, 163],
        "exang": [0, 0, 0, 0, 1],
        "oldpeak": [2.3, 3.5, 1.4, 0.8, 0.6],
        "slope": [0, 0, 2, 2, 2],
        "ca": [0, 0, 0, 0, 0],
        "thal": [1, 2, 2, 2, 2],
        "target": [1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_pipeline():
    """Returns a simple fitted pipeline for testing predictions."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42))
    ])
    # Create dummy data for fitting
    X = np.random.rand(10, 13)
    y = np.random.randint(0, 2, 10)
    pipeline.fit(X, y)
    return pipeline
