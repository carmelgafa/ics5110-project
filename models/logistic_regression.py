from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils.preprocessing_pipeline import preprocessor


logistic_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression())
])