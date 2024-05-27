import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from src.ml.model import inference
from src.ml.data import process_data

import os

# if "DYNO" in os.environ and os.path.isdir(".dvc"):
#     os.system("dvc config core.no_scm true")
#     if os.system("dvc pull") != 0:
#         exit("dvc pull failed")
#     os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Data(BaseModel):
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    age: int
    workclass: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race":"White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                    "age": 39,
                    "workclass": "State-gov"
                }
            ]
        }
    }

with open("./model/model.pkl", "rb") as f:
    model = pickle.load(f)



@app.post("/predict")
async def predict(data: Data):

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
    ]

    with open("./model/model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("./model/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    
    with open("./model/lb.pkl", "rb") as f:
        lb = pickle.load(f)

    X = pd.DataFrame(data.model_dump(by_alias=True), index=[0])
    x_pred, _, _, _ = process_data(X=X, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)
    y_pred = inference(model=model, X=x_pred)
    return {"prediction": y_pred.tolist()}

@app.get("/")
async def welcome():
    return {"message": "Greetings"}