from pydantic import BaseModel
from typing import List

class GestureInput(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    direction: str
