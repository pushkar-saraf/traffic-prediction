# Traffic Prediction with METR-LA Dataset

## Overview
This project involves developing, deploying, and monitoring a traffic prediction model using the METR-LA dataset. The project is divided into three phases: Experimentation, Deployment, and Monitoring.

### Prerequisites
- Python 3.8+
- Docker
- Kubernetes
- Kubeflow Pipelines
- Minikube (for local Kubernetes testing)
- Kubeflow SDK (`pip install kfp`)
- Flask (`pip install flask`)
- Evidently (`pip install evidently`)

## Phase 1: Model Experimentation
1. **Goal**: Train LSTM and GRU models and track them using Kubeflow.
2. **Steps**:
   - Run `starter_notebook_phase_1.py`.
   - Implement your model training pipeline, and track experiments using the Kubeflow Pipelines UI.

## Phase 2: Model Deployment
1. **Goal**: Deploy the trained model using Docker and Kubernetes.
2. **Steps**:
   - Use `starter_notebook_phase_2.py` to set up a Flask API.
   - Create a `Dockerfile` to containerize the API and deploy it using Kubernetes.

## Phase 3: Model Monitoring
1. **Goal**: Monitor the deployed model's performance using Evidently.
2. **Steps**:
   - Run `starter_notebook_phase_3.py` to generate a monitoring dashboard.
   - Check the `evidently_report.html` file for insights on data drift and model performance.

## Useful Links
- [Kubeflow Documentation](https://www.kubeflow.org/docs/)
- [Evidently Documentation](https://docs.evidentlyai.com/)

## How to Submit
- Upload your code, documentation, and deliverables to a GitHub repository.
