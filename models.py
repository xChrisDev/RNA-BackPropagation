from pydantic import BaseModel
from typing import List

class TrainRequest(BaseModel):
    niu: float
    alfa: float
    rms: float
    epochs: int
    upper_limit: float
    lower_limit: float
    hidden_neurons: int
    input_neurons: int
    training_patterns: List[List[float]]

class PredictRequest(BaseModel):
    hidden_neurons: int
    input_neurons: int
    output_neurons: int
    inputs: List[float]
