import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import your prediction function (make sure this matches your prediction script file name)
from src.models.predict_model import predict

# Initialize FastAPI app
app = FastAPI(title="Student Placement Prediction API")

# Ensure a logs directory exists and configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Define the expected input structure using Pydantic
class PredictionRequest(BaseModel):
    data: dict


# 1. Home endpoint to verify the API is running
@app.get("/")
def home():
    return {"message": "ML API running"}


# 2. Main prediction endpoint
@app.post("/predict")
def make_prediction(request: PredictionRequest):
    try:
        # Validate that data is provided
        if not request.data:
            raise HTTPException(
                status_code=400, detail="Invalid input data provided."
            )

        # Call your prediction helper function
        result = predict(request.data)

        # Log the request and result
        logging.info(
            f"Input: {request.data} | Prediction Result: {result}"
        )

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