import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path so we can import train
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import train

def test_load_data(tmp_path):
    """Test data loading functionality."""
    # Create a dummy CSV file
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df_orig = pd.DataFrame(data=d)
    file_path = tmp_path / "test_data.csv"
    df_orig.to_csv(file_path, index=False)
    
    # Load it using the function
    df_loaded = train.load_data(str(file_path))
    
    pd.testing.assert_frame_equal(df_orig, df_loaded)

def test_evaluate():
    """Test the evaluation metrics calculation."""
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_pred_proba = [0.1, 0.9, 0.4, 0.2]
    
    metrics = train.evaluate(y_true, y_pred, y_pred_proba)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 0.75  # 3 correct out of 4

@patch("train.cross_validate")
def test_cross_validate_model(mock_cv, sample_data):
    """Test cross validation wrapper."""
    # Mock return value of cross_validate
    mock_cv.return_value = {
        "test_accuracy": np.array([0.8, 0.9]),
        "test_precision": np.array([0.7, 0.8]),
        "test_recall": np.array([0.6, 0.7]),
        "test_roc_auc": np.array([0.85, 0.95])
    }
    
    pipeline = MagicMock()
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]
    
    metrics = train.cross_validate_model(pipeline, X, y)
    
    assert metrics["cv_accuracy"] == pytest.approx(0.85)
    assert metrics["cv_precision"] == pytest.approx(0.75)

@patch("train.cross_validate_model")
@patch("train.train_test_split")
@patch("train.mlflow")
@patch("train.confusion_matrix")
@patch("train.plt")
@patch("train.sns")
def test_main_rf(mock_sns, mock_plt, mock_cm, mock_mlflow, mock_split, mock_cv_model, sample_data, tmp_path):
    """Smoke test for main function with Random Forest."""
    # Mock data loading
    data_path = tmp_path / "data.csv"
    sample_data.to_csv(data_path, index=False)

    # Mock split return
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]
    mock_split.return_value = (X, X, y, y)
    
    # Mock cross_validate_model return to avoid splitting issues with small data
    mock_cv_model.return_value = {
        "cv_accuracy": 0.8, "cv_precision": 0.7, "cv_recall": 0.6, "cv_roc_auc": 0.9
    }
    
    # Mock args
    args = MagicMock()
    args.data_path = str(data_path)
    args.model_type = "rf"
    args.n_estimators = 10
    args.max_depth = 5
    
    # Mock run context
    mock_run = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_run.info.run_id = "test_run_id"
    mock_run.info.artifact_uri = "file:///tmp/mlruns/0/test_run_id/artifacts"
    
    # Run main
    with patch("train.load_data", return_value=sample_data):
        with patch("mlflow.artifacts.download_artifacts"): # mock download
            train.main(args)
    
    # Verify mlflow calls
    mock_mlflow.set_experiment.assert_called_with("heart_disease_prediction")
    mock_mlflow.log_param.assert_any_call("model_type", "rf")
    mock_mlflow.log_metrics.assert_called()
    mock_mlflow.sklearn.log_model.assert_called()
