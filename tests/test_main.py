from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get_return():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_inference_low_salary():
    response = client.post(
        "/inference",
        json={'age': 39,
              'workclass': 'State-gov',
              'fnlgt': 77516,
              'education': 'Bachelors',
              'education-num': 13,
              'marital-status': 'Never-married',
              'occupation': 'Adm-clerical',
              'relationship': 'Not-in-family',
              'race': 'White',
              'sex': 'Male',
              'capital-gain': 2174,
              'capital-loss': 0,
              'hours-per-week': 40,
              'native-country': 'United-States'}
    )
    assert response.status_code == 200
    result = [int(elem) for elem in list(response.json().strip('[]'))]
    assert result == [0]


def test_inference_high_salary():
    response = client.post(
        "/inference",
        json={'age': 50,
              'workclass': 'Federal-gov',
              'fnlgt': 251585,
              'education': 'Bachelors',
              'education-num': 13,
              'marital-status': 'Divorced',
              'occupation': 'Exec-managerial',
              'relationship': 'Not-in-family',
              'race': 'White',
              'sex': 'Male',
              'capital-gain': 0,
              'capital-loss': 0,
              'hours-per-week': 55,
              'native-country': 'United-States'}
    )
    assert response.status_code == 200
    result = [int(elem) for elem in list(response.json().strip('[]'))]
    assert result == [1]
