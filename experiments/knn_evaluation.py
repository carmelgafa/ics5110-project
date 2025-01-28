import joblib
from models.knn import knn_pipeline
from pipeline_load_data import load_pipeline_data
from pipeline_evaluation import evaluate_pipeline


# Load the data
X_train, y_train, X_test, y_test = load_pipeline_data()

# Train the pipeline
knn_pipeline.fit(X_test, y_test)

# Evaluate the pipeline
evaluate_pipeline('knn', knn_pipeline, X_test, y_test)

# Save the pipeline
joblib.dump(knn_pipeline, 'models/knn_pipeline.pkl')
print("Pipeline saved successfully.")