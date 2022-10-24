# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_metrics_for_slices
from pandas import read_csv
from joblib import dump
#from joblib import load

# Add code to load in the data.
data = read_csv('./data/census_cleaned.csv')

# Split data into training and test sets.
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

# Preprocess the training data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label='salary', encoder=encoder,
    lb=lb, training=False
)

# Train and save a model. Binarizer, encoder and categorical features are 
# required to process new data before feeding it to the classifier.
# Ignore first column which contains the orginal index of each record.
clf = train_model(X_train[:,1:], y_train)
model = {
    'encoder': encoder,
    #'lbl_binarizer': lb,
    'classifier': clf,
    'cat_features': cat_features
}

dump(model, './model/model.joblib')

#model = load('./model/model.joblib')
#X_test, y_test, encoder, lb = process_data(
#    test, categorical_features=model['cat_features'], 
#    encoder=model['encoder'], training=False
#)

# Compute model performance on slices using test data

compute_metrics_for_slices(
    model['classifier'],
    test,
    X_test,
    y_test,
    cat_features
)
