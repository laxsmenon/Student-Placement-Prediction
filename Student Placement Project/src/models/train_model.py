import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_model(n_estimators, random_state, run_name):
  """Trains a Random Forest model and logs parameters/metrics into MLflow."""
  project_dir = Path(__file__).resolve().parents[2]
  data_path = project_dir / "data" / "raw" / "student_placement_data.csv"

  df = pd.read_csv(data_path)

  # Identify target column dynamically
  target_column = "placement"
  if target_column not in df.columns:
    possible_targets = [col for col in df.columns if "place" in col.lower()]
    if possible_targets:
      target_column = possible_targets[0]

  X = df.drop(columns=[target_column])
  y = df[target_column]

  # Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=random_state
  )

  # Setup MLflow Tracking backend
  db_path = project_dir / "mlflow.db"
  mlflow.set_tracking_uri(f"sqlite:///{db_path}")
  mlflow.set_experiment("student placement prediction")

  with mlflow.start_run(run_name=run_name):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(
        f"Run '{run_name}' Complete! n_estimators={n_estimators} | Accuracy:"
        f" {accuracy:.4f}"
    )

    # Log parameters, metrics, and model artifact
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        clf, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
    )


def select_and_save_best_model():
  """Queries MLflow runs, identifies the best performing model, and saves it locally."""
  project_dir = Path(__file__).resolve().parents[2]
  db_path = project_dir / "mlflow.db"
  mlflow.set_tracking_uri(f"sqlite:///{db_path}")

  client = MlflowClient()
  experiment = client.get_experiment_by_name("student placement prediction")
  experiment_id = experiment.experiment_id

  # Search runs sorted by accuracy in descending order
  runs = client.search_runs(
      experiment_ids=[experiment_id],
      order_by=["metrics.accuracy DESC"],
      max_results=1,
  )

  if not runs:
    print("No runs found in MLflow!")
    return

  best_run = runs[0]
  best_run_id = best_run.info.run_id
  best_accuracy = best_run.data.metrics.get("accuracy")

  print(f"\nBest Run Found! Run ID: {best_run_id} | Accuracy: {best_accuracy:.4f}")

  # Load the best model directly from MLflow artifact storage
  model_uri = f"runs:/{best_run_id}/model"
  best_model = mlflow.sklearn.load_model(model_uri)

  # Save the best model locally for FastAPI inference
  models_dir = project_dir / "models"
  os.makedirs(models_dir, exist_ok=True)
  model_save_path = models_dir / "model.pkl"
  joblib.dump(best_model, model_save_path)
  print(f"Best model successfully saved locally to {model_save_path}\n")


if __name__ == "__main__":
  # Define multiple hyperparameter configurations to compare
  configs = [
      {"n_estimators": 50, "random_state": 42, "run_name": "RF_50_Estimators"},
      {"n_estimators": 100, "random_state": 42, "run_name": "RF_100_Estimators"},
      {"n_estimators": 200, "random_state": 42, "run_name": "RF_200_Estimators"},
  ]

  # Execute training runs
  for config in configs:
    train_model(
        n_estimators=config["n_estimators"],
        random_state=config["random_state"],
        run_name=config["run_name"],
    )

  # Automatically pick and save the best model
  select_and_save_best_model()