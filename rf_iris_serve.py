import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Iris Predictor API")

# Load the model and feature names
model_path = os.path.join("artifacts", "random_forest_model.joblib")
features_path = os.path.join("artifacts", "feature_names.joblib")

try:
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
except FileNotFoundError:
    raise Exception("Model files not found. Please run model_trainer.py first.")


# Define the request body model
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Define the response model
class IrisResponse(BaseModel):
    predicted_class: int
    predicted_class_name: str
    probability: float


# Map numeric labels to class names
class_names = ["setosa", "versicolor", "virginica"]


@app.post("/predict", response_model=IrisResponse)
async def predict(request: IrisRequest):
    try:
        # Convert input data to array
        features = [
            [
                request.sepal_length,
                request.sepal_width,
                request.petal_length,
                request.petal_width,
            ]
        ]

        # Make prediction
        prediction = model.predict(features)[0]
        probability = np.max(model.predict_proba(features)[0])

        return IrisResponse(
            predicted_class=int(prediction),
            predicted_class_name=class_names[prediction],
            probability=float(probability),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
