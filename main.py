from typing import List

from fastapi import FastAPI # todo: look into Depends
from pydantic import BaseModel, validator

from ml.model import score

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

app = FastAPI()


@app.post("/predict", response_model=PredictResponse)
def predict(input: PredictRequest, model_type: ModelType='LogisticRegression'):
    y_pred = score(input.data, str(model_type.data))
    return PredictResponse(data=y_pred)
