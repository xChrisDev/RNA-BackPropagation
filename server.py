from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from machine import Machine
from models import TrainRequest, PredictRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/rms")
def get_rms_historic():
    return Machine.data_return()

@app.post("/api/train")
def post_training(data: TrainRequest):
    return Machine.learn(
        training_patterns=data.training_patterns,
        alfa=data.alfa,
        hidden_neurons=data.hidden_neurons,
        input_neurons=data.input_neurons,
        lower_limit=data.lower_limit,
        upper_limit=data.upper_limit,
        max_epochs=data.epochs,
        niu=data.niu,
        output_neurons=3,
        rms=data.rms,
    )

@app.post("/api/predict")
def post_predict(data: PredictRequest):
    return Machine.predict(
        hidden_neurons=data.hidden_neurons,
        input_neurons=data.input_neurons,
        inputs=data.inputs,
        output_neurons=data.output_neurons,
    )
