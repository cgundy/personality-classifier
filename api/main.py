from typing import List

from fastapi import FastAPI # todo: look into Depends
from pydantic import BaseModel, validator

from .ml.model import score, get_accuracy

# Todo: move to config
valid_models = ['LogisticRegression']

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
def predict(text_input: PredictRequest, model_type: ModelType):
    y_pred = score(text_input.data, str(model_type.data))
    return PredictResponse(data=y_pred)

@app.post("/accuracy", response_model=PredictResponse)
def return_accuracy(model_type: ModelType):
	print(get_accuracy(str(model_type.data)))
	accuracy=get_accuracy(str(model_type.data))
	return AccuracyResponse(data=accuracy)
