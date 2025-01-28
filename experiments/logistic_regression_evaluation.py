import joblib
from models.logistic_regression import logistic_pipeline
from pipeline_load_data import load_pipeline_data
from pipeline_evaluation import evaluate_pipeline


X_train, y_train, X_test, y_test = load_pipeline_data()

# Train the logistic pipeline
logistic_pipeline.fit(X_train, y_train)
 
evaluate_pipeline('lr', logistic_pipeline, X_test, y_test)

# Save the pipeline
joblib.dump(logistic_pipeline, 'models/logistic_pipeline.pkl')
print("Pipeline saved successfully.")