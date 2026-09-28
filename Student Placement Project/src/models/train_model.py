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
  # 1. Setup dynamic paths
  project_dir = Path(__file__).resolve().parents[2]
  data_path = project_dir / "data" / "raw" / "student_placement_data.csv"
  models_dir = project_dir / "models"
  os.makedirs(models_dir, exist_ok=True)

  print(f"Loading data from {data_path}...")
  df = pd.read_csv(data_path)

  # Clean column names (replace spaces with underscores if needed for consistency)
  target_column = "placement"  # Update target column name if needed
  if target_column not in df.columns:
    # Fallback check for common target names
    possible_targets = [col for col in df.columns if "place" in col.lower()]
    if possible_targets:
      target_column = possible_targets[0]

  X = df.drop(columns=[target_column])
  y = df[target_column]

  # 2. Train-Test Split
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

    mlflow.sklearn.log_model(
    clf, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    # Save model locally for FastAPI production inference
    model_save_path = models_dir / "model.pkl"
    joblib.dump(clf, model_save_path)
    print(f"Local model saved to {model_save_path}")


if __name__ == "__main__":
  train()