import requests
import json

response = requests.get("https://im-tired-boss2-9c09a0b9b543.herokuapp.com/")
print("====================")
print("GET")
print(f"status code: {response.status_code}")
print(f"Result: {response.text}")
print("====================\n\n")

input_data = \
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
response = requests.post("https://im-tired-boss2-9c09a0b9b543.herokuapp.com/predict",
                         data=json.dumps(input_data))
print("====================")
print("POST")
print(f"status code: {response.status_code}")
print(f"Input: {input_data}")
print(f"Result: {response.text}")
print("====================\n\n")

input_data = \
{
    "fnlgt": 45781,
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
    "workclass": "Private"
}
response = requests.post("https://im-tired-boss2-9c09a0b9b543.herokuapp.com/predict",
                         data=json.dumps(input_data))
print("====================")
print("POST")
print(f"status code: {response.status_code}")
print(f"Input: {input_data}")
print(f"Result: {response.text}")
print("====================\n\n")