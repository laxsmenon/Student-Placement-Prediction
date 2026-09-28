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
