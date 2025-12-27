import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    return pd.read_csv(path)

def evaluate(y_true, y_pred, y_pred_proba):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main(args):
    print("Loading data...")
    df = load_data(args.data_path)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set experiment
    mlflow.set_experiment("heart_disease_prediction")
    
    with mlflow.start_run():
        # Define Pipeline
        if args.model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        else:
            clf = LogisticRegression(random_state=42, max_iter=1000)
            
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        print(f"Training {args.model_type} model...")
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate(y_test, y_pred, y_prob)
        print(f"Metrics: {metrics}")
        
        # Log params
        mlflow.log_param("model_type", args.model_type)
        if args.model_type == 'rf':
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log artifacts
        plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        print("Run complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/processed/heart_disease_cleaned.csv")
    parser.add_argument("--model_type", default="rf", choices=["rf", "lr"])
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    main(args)
