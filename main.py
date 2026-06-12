from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI()

try:
    model = joblib.load('models/model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/")
def read_root():
    return {"status": "Prototype API is running"}

@app.post("/predict")
async def get_prediction():
    try:
        forecast = model.predict(h=30).reset_index()
        
        forecast['ds'] = forecast['ds'].astype(str)
        
        return {"forecast": forecast.to_dict(orient='records')}
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}", flush=True) 
        raise HTTPException(status_code=500, detail=str(e))