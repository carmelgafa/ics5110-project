from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from utils.preprocessing_pipeline import preprocessor
import os

# Define the KNN model



# Load best parameters
parameters_path = 'results/knn_best_params.txt'
best_params = {}

if os.path.exists(parameters_path):
    with open(parameters_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Best Parameters:"):
                best_params = eval(line.split(":", 1)[1].strip())


# knn_classifier = KNeighborsClassifier(
#     n_neighbors=best_params.get('classifier__n_neighbors', 5),
#     weights=best_params.get('classifier__weights', 'uniform'),
#     metric=best_params.get('classifier__metric', 'minkowski'),
#     p=best_params.get('classifier__p', 2)
# )


knn_classifier = KNeighborsClassifier(
    n_neighbors=best_params.get('classifier__n_neighbors', 5),
    weights= 'distance',
    metric=best_params.get('classifier__metric', 'minkowski'),
    p=best_params.get('classifier__p', 2)
)

# Define pipeline
knn_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),  # Reuse preprocessing pipeline
    ('classifier', knn_classifier)
])
