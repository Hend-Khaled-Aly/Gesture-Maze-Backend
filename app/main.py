from fastapi import FastAPI
from app.schemas import GestureInput, PredictionResponse
from app.model import predict_direction

app = FastAPI(title="Gesture Maze Solver")

@app.get("/")
def read_root():
    return {"message": "Gesture Maze Backend is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_gesture(input_data: GestureInput):
    direction = predict_direction(input_data.features)
    return PredictionResponse(direction=direction)
