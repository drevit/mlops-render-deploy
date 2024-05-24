"""
Script to train machine learning model.

Author: Andrea Vitali
Date: May 2024
"""
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from src.ml.data import process_data
from src.ml.model import train_model, compute_model_metrics

data = pd.read_csv("./data/census_clean.csv")
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train=X_train, y_train=y_train)

with open("./model/model.pkl", "wb") as f:
    pickle.dump(model, f)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
prec_train, rec_train, fb_train = compute_model_metrics(y_train, y_train_pred)
prec_test, rec_test, fb_test = compute_model_metrics(y_test, y_test_pred)

print(f"Train metrics:\n\nPrecision: {prec_train}\nRecall: {rec_train}\nFbeta: {fb_train}")
print("\n===================\n")
print(f"Test metrics:\n\nPrecision: {prec_test}\nRecall: {rec_test}\nFbeta: {fb_test}")