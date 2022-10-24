import json
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import inference, load_model
from starter.ml.data import process_data

app = FastAPI()

class Test_Record(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 38,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


# Define a GET on the root giving a welcome message.
@app.get("/")
async def greeting_message():
    return {"greeting": "Hello World!"}


@app.post("/inference")
async def inference_data(record: Test_Record):
    # load model
    model = load_model()
    # create dataframe from Test_Record object "record"
    df = pd.Series(
        record.dict(by_alias=True)
        ).to_frame().transpose().infer_objects()

    # process data
    X, _, _, _ = process_data(
        X=df,
        categorical_features=model['cat_features'],
        encoder=model['encoder'],
        training=False
    )

    # load model and return inference on X
    model = load_model()
    return json.dumps(inference(model['classifier'], X).tolist())
