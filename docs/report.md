# MLOps Experimental Learning Assignment Report

## 1. Introduction
This project implements an end-to-end MLOps pipeline for predicting heart disease risk. It covers data ingestion, exploratory data analysis, model training with experiment tracking, and containerized deployment on Kubernetes.

## 2. Methodology

### 2.1 Data Acquisition & EDA
- **Source**: UCI Heart Disease Dataset (Processed Cleveland).
- **Preprocessing**: 
  - Missing values dropped/imputed.
  - Categorical variables encoded.
  - Target variable binarized (0: No Disease, 1: Disease).
- **EDA**: See `notebooks/01_eda.ipynb` for histograms and correlation heatmaps.

### 2.2 Model Development
- **Algorithms**: Random Forest and Logistic Regression were evaluated.
- **Tracking**: MLflow was used to log parameters (`n_estimators`, `max_depth`) and metrics (`accuracy`, `roc_auc`).
- **Selection**: Random Forest was selected based on higher ROC-AUC scores (approx 0.94).

### 2.3 Reproducibility
- **Dependencies**: Managed via `requirements.txt`.
- **Pipeline**: `src/train.py` ensures consistent data splits and preprocessing via Scikit-Learn Pipelines.
- **Packaging**: Model artifacts are serialized with MLflow and packaged into a Docker container.

## 3. System Architecture
[Mermaid Diagram Placeholder - See Implementation Plan]

The system consists of:
1. **Training Pipeline**: Python script -> MLflow.
2. **Serving**: FastAPI application loading the model artifact.
3. **Infrastructure**: Kubernetes Deployment and LoadBalancer Service.

## 4. CI/CD & Deployment
- **GitHub Actions**:
  - Triggers on push to `main`.
  - Runs linting (flake8) and smoke tests.
  - Builds Docker image and pushes to Docker Hub.
- **Kubernetes**:
  - Deployed using `kubectl apply -f k8s/`.
  - Exposed via LoadBalancer on port 80.

## 5. Monitoring
- **Prometheus**: The API exposes specific metrics at `/metrics` using `prometheus-fastapi-instrumentator`. 
- **Logs**: Standard output logging for request tracing.

## 6. Usage Instructions
1. **Install Requirements**: `pip install -r requirements.txt`
2. **Download Data**: `python scripts/download_data.py`
3. **Train Model**: `python src/train.py`
4. **Run API**: `uvicorn src.api:app --reload`

## 7. Screenshots
*(Placeholders for: Deployment success, MLflow UI, Grafana Dashboard)*
