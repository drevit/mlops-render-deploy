import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.getcwd(), ".."))

@pytest.fixture(scope="session")
def data():
    df = pd.read_csv("./data/census_clean.csv")
    return df

def test_data_length(data):
    assert len(data) > 1

def test_data_width(data):
    assert len(data.columns) == 16