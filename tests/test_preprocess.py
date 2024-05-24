import pytest
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append("..")

from src.ml.data import process_data

@pytest.fixture(scope="session")
def model():
    with open("./model/model.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture(scope="session")
def data():
    df = pd.read_csv("./data/census_clean.csv")
    return df

@pytest.fixture(scope="session")
def cat_features():
    return [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def test_preprocess_output_length(data, cat_features):
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )
    assert len(X_train) > 0
    assert len(y_train) > 0
    X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
    )
    assert len(X_test) > 0
    assert len(y_test) > 0

