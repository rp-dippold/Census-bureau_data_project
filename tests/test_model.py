import pytest

import pandas as pd
import numpy as np

from starter.ml.model import load_model, inference, compute_model_metrics
from starter.ml.data import process_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def inference_data():
    """Generate inference data."""
    df = pd.DataFrame(
        {
            'age': [50, 39],
            'workclass': ['Federal-gov', 'State-gov'],
            'fnlgt': [251585, 77516],
            'education': ['Bachelors', 'Bachelors'],
            'education-num': [13, 13],
            'marital-status': ['Divorced', 'Never-married'],
            'occupation': ['Exec-managerial', 'Adm-clerical'],
            'relationship': ['Not-in-family', 'Not-in-family'],
            'race': ['White', 'White'],
            'sex': ['Male', 'Male'],
            'capital-gain': [0, 2174],
            'capital-loss': [0, 0],
            'hours-per-week': [55, 40],
            'native-country': ['United-States', 'United-States'],
        }
    )
    return df, np.array([1, 0])


def test_load_model_types():
    """Test for function "load model" with respect to the return type,
       the number of dictionary elements, the dictionary's keys and
       the type of the dictionary's values.
    """
    model = load_model()
    assert type(model) == dict
    assert len(model) == 3
    assert set(model.keys()) == set(['cat_features', 'encoder', 'classifier'])
    assert type(model['cat_features']) == list and \
           type(model['encoder']) == OneHotEncoder and \
           type(model['classifier']) == RandomForestClassifier


def test_compute_model_metrics(inference_data):
    """Test for function "compute_model_metrics" regarding the returned
       number of metrics and if the scores are in range 0.0 to 1.0.
    """
    model = load_model()

    X, _, _, _ = process_data(X=inference_data[0],
                              categorical_features=model['cat_features'],
                              training=False,
                              encoder=model['encoder'])

    preds = inference(model['classifier'], X)

    metrics = compute_model_metrics(inference_data[1], preds)
    assert len(metrics) == 3
    assert 0.0 <= metrics[0] <= 1.0 and \
           0.0 <= metrics[1] <= 1.0 and \
           0.0 <= metrics[2] <= 1.0


def test_inference(inference_data):
    """Test for function "inference" with respect to the return type, the
       returned value's shape and the expected predictions.
    """
    model = load_model()

    X, _, _, _ = process_data(X=inference_data[0],
                              categorical_features=model['cat_features'],
                              training=False,
                              encoder=model['encoder'])

    preds = inference(model['classifier'], X)

    assert type(preds) == np.ndarray
    assert preds.shape == (2,)
    assert (preds == inference_data[1]).any()
