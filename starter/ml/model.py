import numpy as np

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import load


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, simple=True):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    if simple:
        clf = RandomForestClassifier(n_estimators = 200,
                                     max_depth = 50,
                                     min_samples_split = 20,
                                     random_state=42)
        return clf.fit(X_train, y_train)

    param_grid = {'n_estimators': [500, 700, 1000],
                  'max_depth': [30, 50, 100],
                  'min_samples_split': [20, 40, 60]}

    base_estimator = RandomForestClassifier(random_state=42)

    clf = GridSearchCV(base_estimator, param_grid, cv=5,
                       scoring='f1', verbose=4)
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def load_model():
    """Load an existing model.
    
    Returns
    -------
    model: dict
        A dictionary containing the classifier 'clf', the
        encoder 'encoder' the label binarizer 'lbl_binarizer'
        and all categorical features 'cat_features'.
    """
    model = load('./model/model.joblib')
    return model


def compute_metrics_for_slices(clf, data, X, y, categories):
    """Compute the metrics of a model for each categorical variable held fixed.

    Inputs
    ------
    clf: sklearn classifier
        Trained machine learning model.
    data: pandas.DataFrame
        Original DataFrame required to select rows according to category
        values.
    X : np.array
        Data used for prediction
    y : np.array
        Known labels, binarized.
    categories: list
        List of categorical features.
    """
    with open('./metrics/slice_output.txt', 'w', encoding='utf-8') as f:
        for cat in categories:
            # Compute model performance on slices of cat
            for val in data[cat].unique():
                f.writelines(f'Category: {cat}, Value: {val}\n')
                f.writelines('Metrics:\n')
                # use column "index" (copy of original index) to select all
                # rows belonging to category "cat".
                idx = data[data[cat] == val]['index'].values
                X_cat = X[np.isin(X[:, 0], idx)]
                y_cat = y[np.isin(X[:, 0], idx)]
                if X_cat.shape[0] == 0:
                    f.writelines('   No values found to calculate metrics.\n')
                else:
                    # ignore first column which contains the index
                    pred = inference(clf, X_cat[:,1:])
                    precision, recall, fbeta = compute_model_metrics(y_cat,
                                                                     pred)
                    f.writelines(f'   - Precision: {precision}\n')
                    f.writelines(f'   - Recall: {recall}\n')
                    f.writelines(f'   - F1 Score: {fbeta}\n')
