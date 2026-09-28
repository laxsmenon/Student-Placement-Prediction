from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd


def get_model_path():
    """Get the dynamic path to the saved model."""
    project_dir = Path(__file__).resolve().parents[2]
    return project_dir / "models" / "model.pkl"


@lru_cache(maxsize=1)
def load_model():
    """Load the model with caching to optimize performance."""
    model_path = get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please train the model first."
        )
    return joblib.load(model_path)


def predict(input_data: dict):
  """Takes input features as a dictionary, converts to DataFrame, and predicts."""
  if not input_data:
    raise ValueError("Input data cannot be empty.")

  # Convert dictionary input to a pandas DataFrame
  df = pd.DataFrame([input_data])

  # Load trained model
  model = load_model()

  # Pass df.values (NumPy array) to bypass feature name checks entirely
  prediction = model.predict(df.values)

  return str(prediction[0])