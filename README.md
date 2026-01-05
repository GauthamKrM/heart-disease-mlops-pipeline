# MLOps Heart Disease Prediction Pipeline

This repository contains an end-to-end MLOps pipeline for a Heart Disease Prediction model. It demonstrates automated CI/CD workflows, experiment tracking with MLflow, containerization with Docker, and deployment on Kubernetes.

## Architecture

The following diagram illustrates the complete flow of the pipeline, from code changes to deployment.

```ascii
                                            +---------------------+
                                            |      MLflow         |
                                         -> | (Experiment Track)  |
                                         |  +---------------------+
                                         |
+-------------+      +-------------------+-----------------+       +------------------+
|   Code      |      |         GitHub Actions              |       |                  |
|  Changes    +----> |        (CI/CD Pipeline)             +-----> |    Docker Hub    |
| (Push/PR)   |      |                                     |       | (Image Registry) |
+-------------+      | 1. Build & Test (Lint, Unit Tests)  |       +--------+---------+
                     | 2. Train Model (Smoke & Prod)       |                |
                     | 3. Build & Push Docker Image        |                |
                     +-------------------------------------+                |
                                                                            |
                                                                            v
                                                                 +------------------+
                                                                 |    Kubernetes    |
                                                                 |     Cluster      |
                                                                 | (Minikube/Cloud) |
                                                                 +------------------+
                                                                 |  - Deployment    |
                                                                 |  - Service       |
                                                                 |  - Monitoring    |
                                                                 +------------------+
```

## Pipeline Workflow

The CI/CD pipeline is defined in `.github/workflows/mlops-pipeline.yml` and consists of two main jobs:

### 1. Build and Test
Triggered on every Push and Pull Request.
- **Environment Setup**: Installs Python 3.9 and dependencies.
- **Linting**: Checks code quality using `flake8`.
- **Smoke Training**: Runs the training script (`src/train.py`) with a small subset/parameters to ensure the code executes without errors.
- **Unit Tests**: Runs `pytest` on the `tests/` directory to validate API responses and training logic.

### 2. Docker Build & Push
Triggered only on pushes to the `main` branch, after `Build and Test` succeeds.
- **Production Training**: Retrains the model on the full dataset, logging metrics and artifacts to MLflow.
- **Artifact Handling**: Extracts the trained model artifact.
- **Containerization**: Builds a Docker image containing the FastAPI app and the trained model.
- **Registry**: Pushes the tagged image to Docker Hub (`gauthamkrm/heart-disease-api:latest`).

## Project Structure

- **`.github/workflows`**: CI/CD configuration files.
- **`k8s/`**: Kubernetes manifests for deployment, service, and monitoring (Prometheus).
- **`notebooks/`**: Jupyter notebooks for Exploratory Data Analysis (EDA).
- **`src/`**: Source code for training (`train.py`) and the FastAPI application (`api.py`).
- **`tests/`**: Unit tests for the API and training modules.

## Usage & Commands

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Model**
   ```bash
   # Downloads data and trains the model (saved to models/model)
   python scripts/download_data.py
   python src/train.py --n_estimators 100
   ```

3. **Run API Locally**
   ```bash
   # Ensure model is present in models/model
   python src/api.py
   # API will be available at http://0.0.0.0:8000
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   ```

### Kubernetes Deployment (Minikube)

Prerequisite: Ensure you have `kubectl` and `minikube` installed and running.

1. **Apply Configurations**
   ```bash
   kubectl apply -f k8s/
   ```
   This will create:
   - `heart-disease-api` Deployment (running the Docker image)
   - `heart-disease-service` (LoadBalancer exposing port 80)
   - `prometheus` Deployment & Service (for monitoring)

2. **Access the Service**
   On Minikube, you might need to use `minikube service` to access LoadBalancers:
   ```bash
   minikube service heart-disease-service --url
   minikube service grafana-service --url
   minikube service prometheus-service --url
   ```

### Monitoring via Prometheus

The application is instrumented with `prometheus-fastapi-instrumentator`.
- Metrics are exposed at `/metrics`.
- Prometheus scrapes these metrics via the configuration in `k8s/monitoring.yaml`.

## Experiment Tracking

Model training runs are tracked using MLflow.
- **Metrics**: Accuracy, Precision, Recall, ROC AUC.
- **Parameters**: Model type (Random Forest/Logistic Regression), Hyperparameters.
- **Artifacts**: Confusion Matrix, Trained Model (sklearn format).
