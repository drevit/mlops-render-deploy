import sys
import os
sys.path.append(os.path.join(os.getcwd(), ".."))

import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"] == "Greetings"

def test_post_0():
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
    r = client.post("/predict", content=json.dumps(data))
    
    assert r.status_code == 200
    assert len(r.json().keys()) == 1
    assert r.json()["prediction"] == [0]

def test_post_1():
    data={"fnlgt": 45781,
          "education": "Masters",
          "education-num": 14,
          "marital-status": "Never-married",
          "occupation": "Prof-specialtyl",
          "relationship": "Not-in-family",
          "race": "White",
          "sex": "Female",
          "capital-gain": 14084,
          "capital-loss": 0,
          "hours-per-week": 50,
          "native-country": "United-States",
          "age": 31,
          "workclass": "Private"}

    r = client.post("/predict", content=json.dumps(data))

    assert r.status_code == 200
    assert len(r.json().keys()) == 1
    assert r.json()["prediction"] == [1]