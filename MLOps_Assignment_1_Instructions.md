# Machine Learning Operations (MLOps) — AIMLCZG523
## Assignment 01: End-to-End ML Model Development, CI/CD, and Production Deployment
**Total Marks:** 50

---

## 1. Overview & Objective

### Objective
Design, develop, and deploy a scalable, reproducible machine learning solution using modern MLOps best practices. The assignment emphasizes practical automation, experiment tracking, CI/CD pipelines, containerization, cloud deployment, and monitoring, mirroring real-world production scenarios.

### Problem Statement
Build a machine learning classifier to predict the risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.

### Dataset Details
* **Dataset Title:** Heart Disease UCI Dataset
* **Source:** UCI Machine Learning Repository
* **Format:** CSV containing 14+ features (age, sex, blood pressure, cholesterol, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression, etc.) and a binary target (presence/absence of heart disease).

---

## 2. Assignment Tasks & Mark Breakdown

| Task # | Task Description | Marks |
| :--- | :--- | :---: |
| **Task 1** | **Data Acquisition & Exploratory Data Analysis (EDA)** | **5** |
| **Task 2** | **Feature Engineering & Model Development** | **8** |
| **Task 3** | **Experiment Tracking** | **5** |
| **Task 4** | **Model Packaging & Reproducibility** | **7** |
| **Task 5** | **CI/CD Pipeline & Automated Testing** | **8** |
| **Task 6** | **Model Containerization** | **5** |
| **Task 7** | **Production Deployment** | **7** |
| **Task 8** | **Monitoring & Logging** | **3** |
| **Task 9** | **Documentation & Reporting** | **2** |
| **Total** | | **50** |

---

## 3. Detailed Task Requirements

### Task 1: Data Acquisition & Exploratory Data Analysis (EDA) [5 Marks]
* Obtain the dataset (provide download script or step-by-step instructions).
* Clean and preprocess the data (handle missing values, encode categorical features where applicable).
* Perform Exploratory Data Analysis (EDA) with professional visualizations:
  * Histograms of numeric features
  * Correlation heatmaps
  * Target class balance countplots / distributions

### Task 2: Feature Engineering & Model Development [8 Marks]
* Prepare final ML features (scaling, normalization, encoding).
* Build and train at least two classification models (e.g., Logistic Regression and Random Forest).
* Document model selection criteria and hyperparameter tuning process.
* Evaluate models using cross-validation and relevant evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

### Task 3: Experiment Tracking [5 Marks]
* Integrate **MLflow** (or similar tool) for experiment tracking.
* Log hyperparameters, evaluation metrics, model artifacts, and evaluation plots for all experiment runs.

### Task 4: Model Packaging & Reproducibility [7 Marks]
* Save the final trained model in a reusable format (e.g., MLflow model format, `joblib`/`pickle`, ONNX).
* Write a clean `requirements.txt` (or Conda `environment.yml` file).
* Provide reproducible preprocessing pipelines / transformers (`ColumnTransformer`, `Pipeline`) to ensure end-to-end reproducibility.

### Task 5: CI/CD Pipeline & Automated Testing [8 Marks]
* Write unit tests for data processing and model inference code using `pytest` or `unittest`.
* Create a **GitHub Actions** (or Jenkins) CI/CD pipeline that includes:
  * Code linting / formatting checks
  * Automated unit test execution
  * Model training verification / build step
  * Logging and artifact preservation for each workflow run

### Task 6: Model Containerization [5 Marks]
* Build a Docker container for the model serving API (FastAPI or Flask recommended).
* Expose a `/predict` endpoint that:
  * Accepts JSON request payload
  * Returns predicted class and prediction confidence / probability
* Ensure container builds and runs locally with sample input validation.

### Task 7: Production Deployment [7 Marks]
* Deploy the Dockerized API to a public cloud or local Kubernetes cluster (GKE, EKS, AKS, or Minikube / Docker Desktop).
* Use Kubernetes deployment manifests (`deployment.yaml`, `service.yaml`) or Helm charts.
* Expose the service via LoadBalancer, NodePort, or Ingress.
* Verify endpoints and provide deployment proof/screenshots.

### Task 8: Monitoring & Logging [3 Marks]
* Integrate structured logging of incoming API requests and prediction outputs.
* Demonstrate simple monitoring (e.g., Prometheus + Grafana, or API metrics / logs dashboard).

### Task 9: Documentation & Reporting [2 Marks]
* Submit a professional Markdown or PDF report including:
  * Setup and installation instructions
  * EDA and modeling decisions/justifications
  * Experiment tracking summary and comparisons
  * System architecture diagram
  * CI/CD and deployment workflow screenshots
  * Link to public/accessible code repository

---

## 4. Deliverables

* **GitHub Repository** containing:
  * Source code, `Dockerfile`, `requirements.txt` / `environment.yml`
  * Cleaned dataset and automated download script/instructions
  * Jupyter notebooks & scripts (EDA, training pipeline, inference API)
  * `tests/` directory with automated unit tests
  * GitHub Actions workflow YAML (`.github/workflows/ci-cd.yml`) or `Jenkinsfile`
  * Kubernetes deployment manifests / Helm charts
  * Screenshots folder for reports
* **Final Written Report:** ~10 pages as a `.doc` / `.docx` / `.pdf` file.
* **Deployed API URL** (if publicly hosted) or local access instructions and verification curl commands.
* **Video Recording:** Short video walkthrough demonstrating the overall pipeline (training, CI/CD run, container launch, API test).

---

## 5. Production-Readiness Requirements

1. All scripts must execute seamlessly from a clean setup using the provided `requirements.txt`.
2. The model must serve predictions correctly in an isolated environment (Docker container build & test proof required).
3. The CI/CD pipeline must fail gracefully on code errors or failing tests, outputting clear, actionable logs.

---

## 6. Academic Integrity Note

* Students are expected to complete this assignment independently and demonstrate their own understanding of MLOps concepts and implementation practices.
* Direct copying of code, notebooks, visualizations, feature engineering steps, CI/CD workflows, deployment configurations, or overall project pipelines from other students or online sources is strictly prohibited.
* Submissions exhibiting highly similar implementations (including identical feature selections, EDA visualizations, model pipelines, Docker/Kubernetes configurations, API structures, or GitHub workflows) may be treated as copied submissions subject to academic integrity review and grading penalties.
* Students are encouraged to personalize their approach, experimentation process, architecture decisions, monitoring setup, and documentation.
