from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title = "Disease Prediction API")

try:
    model = joblib.load('disease_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    print(f"Error loading model files: {e}")
    
class PatientData(BaseModel):
    fever: int
    headache: int
    nausea: int
    vomiting: int
    fatigue: int
    joint_pain: int
    skin_rash: int
    cough: int
    weight_loss: int
    yellow_eyes: int
    
@app.get("/")
def home():
    return {"message": "Disease Prediction API is running."}

@app.post("/predict")
def predict_disease(data: PatientData):
    try:
        input_data = [
            data.fever,
            data.headache,
            data.nausea,
            data.vomiting,
            data.fatigue,
            data.joint_pain,
            data.skin_rash,
            data.cough,
            data.weight_loss,
            data.yellow_eyes
        ]
        input_array = np.array(input_data).reshape(1, -1)
        scaled_data = scaler.transform(input_array)
        prediction_index = model.predict(scaled_data)[0]
        
        probs = model.predict_proba(scaled_data)
        confidence = float(np.max(probs))
        
        disease_name = label_encoder.inverse_transform([prediction_index])[0]
        
        return {
            "prediction_id": int(prediction_index),
            "disease_name": disease_name,
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))