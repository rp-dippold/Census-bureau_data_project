import requests
import json

data = {
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

URL = 'https://census-bureau-data-project.herokuapp.com/inference'
response = requests.post(URL, data=json.dumps(data))

print(response.status_code)
print(response.json())