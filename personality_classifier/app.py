from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, validator
from .ml.model_pipeline import predict, get_accuracy
from .ml.utilities import get_config


config = get_config()
valid_models = config["valid_models"]


class PredictRequest(BaseModel):
    data: List[str]


class PredictResponse(BaseModel):
    data: str


class ModelType(BaseModel):
    data: str

    @validator("data")
    def check_model_type(cls, v):
        if v not in valid_models:
            raise ValueError(f"Model type must be one of the following: {valid_models}")
        return v


class AccuracyResponse(BaseModel):
    data: float


app = FastAPI()


@app.post("/predict", response_model=PredictResponse)
def request_prediction(text_input: PredictRequest, model_type: ModelType):
    y_pred = predict(text_input.data, str(model_type.data))[0]
    return PredictResponse(data=y_pred)


@app.get("/accuracy/{model_type}", response_model=AccuracyResponse)
def request_accuracy(model_type: str):
    accuracy = get_accuracy(str(model_type))
    return AccuracyResponse(data=accuracy)
