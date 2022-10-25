# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import compute_metrics_for_slices, compute_model_metrics
from ml.model import train_model, inference, save_model
from pandas import read_csv

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
clf = train_model(X_train[:, 1:], y_train, hyper_tune=False)

# Obtain and save model metrics
preds = inference(clf, X_test[:, 1:])
precision, recall, f1 = compute_model_metrics(y_test, preds)
with open('./metrics/model_metrics.txt', 'w', encoding='utf-8') as f:
    f.writelines('Model Metrics:\n')
    f.writelines(f'   - Precision: {precision}\n')
    f.writelines(f'   - Recall: {recall}\n')
    f.writelines(f'   - F1 Score: {f1}\n')

# Save the classifier, the encoder and the categorical features
# as they are required to process new data before feeded to the classifier.
save_model(clf, encoder, cat_features)

# Compute model performance on slices using test data
compute_metrics_for_slices(
    clf,
    test,
    X_test,
    y_test,
    cat_features
)
