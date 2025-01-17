from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from utils.preprocessing_pipeline import preprocessor

# Define the KNN model
def create_knn_pipeline(n_neighbors=5, weights='uniform', metric='minkowski', p=2):
    """
    Creates a pipeline with preprocessing and a KNN classifier.
    Args:
        n_neighbors (int): Number of neighbors to use.
        weights (str): Weight function used in prediction ('uniform', 'distance').
        metric (str): The metric to use for distance computation.
        p (int): Power parameter for the Minkowski metric.
    Returns:
        Pipeline: A Scikit-learn pipeline.
    """
    knn_classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p
    )

    # Define pipeline
    knn_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),  # Reuse preprocessing pipeline
        ('classifier', knn_classifier)
    ])

    return knn_pipeline
