import joblib
from models.neural_network import nn_pipeline  # Import the neural network pipeline
from pipeline_load_data import load_pipeline_data
from pipeline_evaluation import evaluate_pipeline

# Load the data
X_train, y_train, X_test, y_test = load_pipeline_data()

# Train the logistic pipeline
nn_pipeline.fit(X_train, y_train)

# Evaluate the pipeline
evaluate_pipeline('nn', nn_pipeline, X_test, y_test)

# Save the pipeline
joblib.dump(nn_pipeline, 'models/nn_pipeline.pkl')
print("Pipeline saved successfully.")