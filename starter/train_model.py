# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_metrics_for_slices
from pandas import read_csv
from joblib import dump

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

# Train the model and ignore the first column which contains the orginal
# index of each record. The index is required in "compute_metrics_for_slices"
# to select data belonging to data slices.
clf = train_model(X_train[:, 1:], y_train)

# Save the classifier, the encoder and the categorical features
# as they are required to process new data before feeded to the classifier.
model = {
    'encoder': encoder,
    'classifier': clf,
    'cat_features': cat_features
}
dump(model, './model/model.joblib')

# Compute model performance on slices using test data
compute_metrics_for_slices(
    model['classifier'],
    test,
    X_test,
    y_test,
    cat_features
)
