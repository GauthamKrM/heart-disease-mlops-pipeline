import argparse
import shutil
import os
import tempfile
import pandas as pd
import mlflow
import mlflow.sklearn

# -------------------- Scikit-learn imports --------------------
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# -------------------- Visualization imports --------------------
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    """
    Load dataset from the specified CSV file path.
    """
    return pd.read_csv(path)


def evaluate(y_true, y_pred, y_pred_proba):
    """
    Compute evaluation metrics on test data.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba)
    }


def cross_validate_model(pipeline, X, y):
    #Perform stratified k-fold cross-validation to evaluate model robustness.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Metrics to evaluate during cross-validation
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    # Return mean cross-validation metrics
    return {
        "cv_accuracy": cv_results["test_accuracy"].mean(),
        "cv_precision": cv_results["test_precision"].mean(),
        "cv_recall": cv_results["test_recall"].mean(),
        "cv_roc_auc": cv_results["test_roc_auc"].mean()
    }

# Generate and save a confusion matrix heatmap.
def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main(args):
    print("Loading data...")
    df = load_data(args.data_path)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-test split with stratification to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set MLflow experiment for tracking runs
    mlflow.set_experiment("heart_disease_prediction")

    with mlflow.start_run() as run:

        # ---- Dataset metadata logging (MLOps requirement) ----
        mlflow.log_param("dataset_path", args.data_path)
        mlflow.log_param("num_rows", df.shape[0])
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("random_state", 42)

        # ---- Model selection ----
        if args.model_type == "rf":
            clf = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=42
            )
        else:
            clf = LogisticRegression(random_state=42, max_iter=1000)
        
        # Build preprocessing + model pipeline
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])

        # ---- Cross-validation evaluation  ----
        print("Running cross-validation...")
        cv_metrics = cross_validate_model(pipeline, X_train, y_train)
        mlflow.log_metrics(cv_metrics)

        # ---- Final training ----
        print(f"Training {args.model_type} model...")
        pipeline.fit(X_train, y_train)

        # ---- Test evaluation ----
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # ---- Log parameters ----
        mlflow.log_param("model_type", args.model_type)
        if args.model_type == "rf":
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)

        # ---- Confusion matrix artifact ----
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = os.path.join(tmp_dir, "confusion_matrix.png")
            plot_confusion_matrix(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path)

        # ---- Log model ----
        mlflow.sklearn.log_model(pipeline, "model")

        # ---- Export model locally ----
        local_model_path = "models/model"
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)

        run_id = run.info.run_id
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path="models"
        )

        print("Training completed successfully.")


if __name__ == "__main__":

    # Argument parsing for flexible experiment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/processed/heart_disease_cleaned.csv")
    parser.add_argument("--model_type", default="rf", choices=["rf", "lr"])
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)

    args = parser.parse_args()
    main(args)
