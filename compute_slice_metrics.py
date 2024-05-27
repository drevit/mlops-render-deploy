import sys
import pickle
import pandas as pd
from src.ml.slicing import compute_slice_metrics

df = pd.read_csv("./data/census_clean.csv", index_col=0)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

with open("./model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("./model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("./model/lb.pkl", "rb") as f:
    lb = pickle.load(f)

slice_feature = "education"
label = "salary"
sys.stdout = open("./slice_output.txt", "wt")
compute_slice_metrics(df, cat_features, slice_feature, label, model, encoder, lb)