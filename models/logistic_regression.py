import os

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils.preprocessing_pipeline import preprocessor

logistic_pipeline = None


parameters_path = 'results/logistic_regression_best_params.txt'
if os.path.exists(parameters_path):
    with open(parameters_path, 'r') as f:
        best_params = eval(f.read().strip())

        # Strip 'classifier__' prefix
        stripped_params = {key.split('__')[1]: value for key, value in best_params.items() if key.startswith('classifier__')}

        if stripped_params:
            print("Using best parameters:", stripped_params)
            logistic_pipeline = Pipeline(steps=[
                ('preprocessing', preprocessor),
                ('classifier', LogisticRegression(**stripped_params))
            ])


if logistic_pipeline is None:
    logistic_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression())
    ])
