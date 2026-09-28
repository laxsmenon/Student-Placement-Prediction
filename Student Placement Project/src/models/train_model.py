from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train():
    # 1. Identify project root directory dynamically
    project_dir = Path(__file__).resolve().parents[2]

    # 2. Load dataset from the raw data directory
    # Replace 'placement.csv' with your actual dataset filename if needed
    data_path = project_dir / "data" / "raw" / "student_placement_data.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 3. Split into features (X) and target (y)
    # Assuming 'placement' is the target column name. Change if yours is different.
    target_column = "Placement"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 4. Initialize and train the Random Forest Classifier
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # 5. Create models directory if it doesn't exist and save the model
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")


if __name__ == "__main__":
    train()