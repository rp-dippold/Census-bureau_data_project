# Model Card

## Model Details

R. Peter Dippold created the model. It is a random forest classifier using the default hyperparameters in scikit-learn 1.1.2 except the following parameters which were tuned through hyperparameter_tuning: `n_estimators`, `max_depth`, `min_samples_split`. Parameter
`randam_state` was set to 42.

## Intended Use

This model should be used to predict if a person makes over 50K per year based on census data collected in 1994. Users could be, among others, social or political scientists.

## Training Data

The [census income data](https://archive.ics.uci.edu/ml/datasets/census+income) was obtained from the UCI Machine Learning Repository . The data was cleaned by elminating all leading and trailing white spaces in the data and also in column names.

The original data set has 32561 rows and 15 columns. The content of the target column `salary` is either "<=50K" or ">50K". These values were encoded as 0 and 1, respectively, using a label binarizer. Categorical columns were one-hot encoded. A 80-20 split was used to break the data into a train and test set. No stratification was done.

The original row index was inserted as a separate column to support performance testing on slices.

## Evaluation Data

The model was evaluated on the test set after training. As described above, the test data is a 20 % subset of the original data created during a train-test spilt without startification.

## Metrics

During training F1 score was used as metric. The performance of the final model was evaluated on the test set using F1 Score, Precisiong and Recall. Results for the best model are:

- F1 Score: 0.70
- Precision: 0.78
- Recall: 0.63

## Ethical Considerations

As the dataset contains personal data and it is not checked for data bias, conclusion drawn from the model should be considered with caution.

## Caveats and Recommendations

As the data is from 1994 and therefore outdated, the model should be re-trained on more up to date data before usage.
