import sys
import os
sys.path.append(os.path.join(os.getcwd(), ".."))

import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get():
    r = client.get("/")
    print(r.json())

def test_post():
    data={"fnlgt": 77516,
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
          "native-country": "United-States",
          "age": 39,
          "workclass": "State-gov"}
    r = client.post("/predict", data=json.dumps(data))
    print(r.json())

test_post()