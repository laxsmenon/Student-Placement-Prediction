## 1. Environment Setup & Configuration

### Prerequisites
- **Operating System:** Windows
- **Shell:** PowerShell
- **Runtime:** Python 3.13

### Step-by-Step Environment Initialization
1. **Navigate to Project Directory:**
   ```powershell
   cd C:\Users\laksh\Desktop\mlops\Student Placement Project
   ```

2. **Create a Python Virtual Environment:**
   ```powershell
   python -m venv stdproject
   ```

3. **Activate the Virtual Environment:**
   ```powershell
   .\stdproject\Scripts\Activate
   ```

4. **Install Required Core Dependencies:**
   ```powershell
   pip install fastapi uvicorn scikit-learn pandas joblib pydantic
   ```
   *(Note: The installable package is named `scikit-learn`, while the Python import module is `sklearn`).*

---

## 2. Project Directory Structure

Ensure your project structure follows a standard modular MLOps layout:

```text
Student Placement Project/
│
├── data/
│   └── raw/
│       └── student_placement_data.csv
│
├── models/
│   └── model.pkl
│
├── logs/
│   └── predictions.log
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   └── models/
│       ├── __init__.py
│       ├── train_model.py
│       └── predict_model.py
│
└── stdproject/
```

---

## 3. Data & Training Pipeline (`src/models/train_model.py`)

The training script reads raw tabular data, separates features and target labels, fits a machine learning classifier, and serializes the artifact using `joblib`.

### Implementation Code:
```python
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train():
  # Define paths
  project_dir = Path(__file__).resolve().parents[2]
  data_path = project_dir / "data" / "raw" / "student_placement_data.csv"
  model_dir = project_dir / "models"
  model_dir.mkdir(parents=True, exist_ok=True)
  model_path = model_dir / "model.pkl"

  print(f"Loading data from {data_path}...")
  df = pd.read_csv(data_path)

  # Define features and target column (adjust target column name as per dataset)
  target_column = "placement"  # Change to your dataset's exact target column name if different
  X = df.drop(columns=[target_column])
  y = df[target_column]

  # Train Random Forest Model
  print("Training Random Forest Classifier...")
  model = RandomForestClassifier(random_state=42)
  model.fit(X, y)

  # Save trained model artifact
  joblib.dump(model, model_path)
  print(f"Model successfully saved to {model_path}")


if __name__ == "__main__":
  train()
```

### Execution Command:
```powershell
python src\models\train_model.py
```

---

## 4. Prediction Logic & Feature Resilience (`src/models/predict_model.py`)

The prediction module handles model loading (with optimization/caching) and transforms incoming dictionary payloads into a format consumable by scikit-learn. To prevent runtime failures due to feature name discrepancies (e.g., underscores vs. spaces), column names are automatically sanitized.

### Implementation Code:
```python
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd


def get_model_path():
  project_dir = Path(__file__).resolve().parents[2]
  return project_dir / "models" / "model.pkl"


@lru_cache(maxsize=1)
def load_model():
  model_path = get_model_path()
  if not model_path.exists():
    raise FileNotFoundError(f"Trained model not found at {model_path}.")
  return joblib.load(model_path)


def predict(input_data: dict):
  if not input_data:
    raise ValueError("Input data cannot be empty.")

  # Convert request dictionary into DataFrame
  df = pd.DataFrame([input_data])

  # Automatically resolve spacing/underscore naming differences
  df.columns = df.columns.str.replace("_", " ")

  model = load_model()

  # Generate prediction
  prediction = model.predict(df)
  return str(prediction[0])
```

---

## 5. FastAPI Inference Server (`src/api.py`)

The API exposes endpoints for health checking (`/`) and real-time inference (`/predict`), complete with robust exception handling and automated logging.

### Implementation Code:
```python
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.predict_model import predict

# Initialize FastAPI application
app = FastAPI(title="Student Placement Prediction API")

# Ensure logs directory exists and set up logging configuration
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class PredictionRequest(BaseModel):
  data: dict


@app.get("/")
def home():
  return {"message": "ML API running"}


@app.post("/predict")
def make_prediction(request: PredictionRequest):
  try:
    if not request.data:
      raise HTTPException(
          status_code=400, detail="Invalid input data provided."
      )

    result = predict(request.data)
    logging.info(f"Input: {request.data} | Prediction Result: {result}")
    return {"prediction": result}

  except FileNotFoundError:
    logging.error("Trained model file not found.")
    raise HTTPException(
        status_code=503,
        detail="Model not found. Please train the model first.",
    )
  except Exception as e:
    logging.error(f"Error during prediction: {str(e)}")
    raise HTTPException(status_code=400, detail=str(e))
```

---

## 6. Running and Testing the Service

### Starting the Uvicorn Server:
```powershell
uvicorn src.api:app --reload
```

### Validation via Postman / REST Client:
- **Health Check (GET):**
  - **URL:** `http://127.0.0.1:8000/`
  - **Expected Response:** `{"message": "ML API running"}`

- **Prediction Endpoint (POST):**
  - **URL:** `http://127.0.0.1:8000/predict`
  - **Headers:** `Content-Type: application/json`
  - **Body (raw JSON):**
    ```json
    {
      "data": {
        "CGPA": 7.5,
        "IQ": 120,
        "Profile Score": 80
      }
    }
    ```
  - **Expected Response:** `{"prediction": "Yes"}`

---

## 7. Version Control & Git Workflow

To commit and push your completed MLOps project to GitHub:

1. **Check Status:**
   ```powershell
   git status
   ```
2. **Stage Files:**
   ```powershell
   git add .
   ```
3. **Commit Changes:**
   ```powershell
   git commit -m "Complete MLOps pipeline setup, training module, FastAPI app, and prediction logging"
   ```
4. **Push to Remote Repository:**
   ```powershell
   git push origin main
   ```

# 📊 MLflow Experiment Tracking & Management Guide

This document outlines how the basic student placement model training script was upgraded into a production-grade, tracked MLOps experiment pipeline using **MLflow** and a **SQLite backend database**.

---

## 🛠️ Key Architectural Enhancements

1. **Persistent SQLite Backend Store (`mlflow.db`)**: 
   - Instead of tracking runs in a volatile local folder, the tracking URI is explicitly configured to use a local SQLite database (`sqlite:///mlflow.db`). This ensures that experiment metadata, parameters, and metrics are saved persistently and remain accessible across sessions.
2. **Train-Test Evaluation Split**: 
   - The dataset is split into training and testing sets using `train_test_split` (with a 20% test split), allowing proper evaluation via `accuracy_score` rather than evaluating on the training set.
3. **Artifact Logging & Security Compliance (`skops`)**: 
   - Trained model binaries are saved locally (`models/model.pkl`) for real-time FastAPI inference and logged directly as MLflow artifacts. 
   - Modern security constraints regarding scikit-learn tree structures (`sklearn.tree._tree.Tree`) are addressed by supplying `skops_trusted_types=["sklearn.tree._tree.Tree"]` during artifact logging.

---

## 💻 Code Implementation: `src/models/train_model.py`

Below is the complete, updated training script incorporating MLflow tracking, parameter logging, metric evaluation, and artifact storage:

```python
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train():
  # 1. Setup dynamic project paths
  project_dir = Path(__file__).resolve().parents[2]
  data_path = project_dir / "data" / "raw" / "student_placement_data.csv"
  models_dir = project_dir / "models"
  os.makedirs(models_dir, exist_ok=True)

  print(f"Loading data from {data_path}...")
  df = pd.read_csv(data_path)

  # Identify target column dynamically or fallback
  target_column = "placement"
  if target_column not in df.columns:
    possible_targets = [col for col in df.columns if "place" in col.lower()]
    if possible_targets:
      target_column = possible_targets[0]

  X = df.drop(columns=[target_column])
  y = df[target_column]

  # 2. Train-Test Split for robust evaluation
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # 3. Setup MLflow Tracking with local SQLite store
  db_path = project_dir / "mlflow.db"
  mlflow.set_tracking_uri(f"sqlite:///{db_path}")
  mlflow.set_experiment("student placement prediction")

  n_estimators = 100
  random_state = 42

  # 4. Start MLflow Run
  with mlflow.start_run():
    print("Training RandomForest model...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    clf.fit(X_train, y_train)

    # Evaluate Model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete! Accuracy: {accuracy:.4f}")

    # Log Parameters, Metrics, and Model Artifacts to MLflow
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("accuracy", accuracy)

    # Log model artifact with trusted types handling
    mlflow.sklearn.log_model(
        clf, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # Save local copy for production FastAPI inference
    model_save_path = models_dir / "model.pkl"
    joblib.dump(clf, model_save_path)
    print(f"Local model saved to {model_save_path}")


if __name__ == "__main__":
  train()
```

---

## 🚀 Execution & UI Dashboard Workflow

### Step 1: Execute the Training & Tracking Pipeline
Run the training script from your terminal within your active virtual environment:
```powershell
python src\models\train_model.py
```
*This will execute training, record parameters/metrics into `mlflow.db`, log the model artifact, and serialize `model.pkl`.*

### Step 2: Launch the MLflow Tracking Dashboard
Start the local MLflow server referencing your SQLite backend database:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Step 3: Explore and Compare Runs
1. Open your web browser and navigate to: **`http://127.0.0.1:5000`**
2. Select the **`student placement prediction`** experiment from the sidebar.
3. Review individual run details, compare hyperparameters (`n_estimators`, `random_state`), analyze evaluation metrics (`accuracy`), and inspect the stored model artifacts.
  ---

## 🔄 Multi-Run Experiment Comparison & Automated Best Model Selection

To transition from single-run tracking to a production-grade MLOps workflow, the training pipeline is upgraded to execute **multiple hyperparameter configurations**, log each variant to MLflow, and programmatically select the best-performing model using the **`MlflowClient`**.

---

### 🧠 How It Works

1. **Parameterized Training Function**: 
   - The training logic accepts hyperparameters (e.g., `n_estimators`, `random_state`) and a custom `run_name` as parameters, allowing it to execute iteratively across different configurations.
2. **Systematic Tracking**: 
   - Each run logs its parameters, evaluation metrics (`accuracy`), and model artifact independently into the local SQLite backend (`mlflow.db`).
3. **Automated Best Model Selection (`MlflowClient`)**: 
   - Instead of manually guessing which model is optimal, an automated query uses `MlflowClient.search_runs()` to filter the experiment, sort the runs by accuracy in descending order (`metrics.accuracy DESC`), and retrieve the top-performing run.
4. **Production Artifact Generation**: 
   - The best model is dynamically loaded straight from MLflow's artifact store using its unique `run_id` (`runs:/{run_id}/model`) and serialized locally to `models/model.pkl` for FastAPI inference.

---

### 💻 Updated Implementation Summary (`src/models/train_model.py`)

```python
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn
# ... other imports ...

def train_model(n_estimators, random_state, run_name):
    # Trains model, logs parameters/metrics, and records artifact under run_name
    ...

def select_and_save_best_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("student placement prediction")
    
    # Fetch top run sorted by accuracy descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    
    # Load best model from MLflow and save locally for FastAPI inference
    model_uri = f"runs:/{best_run_id}/model"
    best_model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(best_model, "models/model.pkl")

if __name__ == "__main__":
    configs = [
        {"n_estimators": 50, "random_state": 42, "run_name": "RF_50_Estimators"},
        {"n_estimators": 100, "random_state": 42, "run_name": "RF_100_Estimators"},
        {"n_estimators": 200, "random_state": 42, "run_name": "RF_200_Estimators"},
    ]
    for config in configs:
        train_model(**config)
    
    select_and_save_best_model()


```
---
###📊 Visualizing and Comparing Runs in MLflow UI
##Run the multi-run script:
python src/models/train_model.py
##Launch the dashbaord
mlflow ui --backend-store-uri sqlite:///mlflow.db
Launch the dashboard:

Bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
Open http://127.0.0.1:5000, select your experiment, and use the UI side-by-side comparison feature to analyze how varying the n_estimators parameter impacts model accuracy.
