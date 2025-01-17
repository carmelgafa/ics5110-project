import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from models.knn import create_knn_pipeline

# Load test data
test_data = pd.read_csv('data/test/test_compas-scores-two-years.csv')
X_test = test_data.drop(columns=['two_year_recid'])
y_test = test_data['two_year_recid']

# Load best parameters
parameters_path = 'results/knn_best_params.txt'
best_params = {}

if os.path.exists(parameters_path):
    with open(parameters_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Best Parameters:"):
                best_params = eval(line.split(":", 1)[1].strip())

# Create pipeline with best parameters
knn_pipeline = create_knn_pipeline(
    n_neighbors=best_params.get('classifier__n_neighbors', 5),
    weights=best_params.get('classifier__weights', 'uniform'),
    metric=best_params.get('classifier__metric', 'minkowski'),
    p=best_params.get('classifier__p', 2)
)

# Train the pipeline
knn_pipeline.fit(X_test, y_test)

# Evaluate the pipeline
print("Evaluating the KNN pipeline...")
y_pred = knn_pipeline.predict(X_test)
y_pred_proba = knn_pipeline.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print(f"Test Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Save metrics
with open('results/knn_evaluation_metrics.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_pipeline.classes_)
disp.plot()
plt.savefig('results/knn_confusion_matrix.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
plt.figure()
plt.plot(fpr, tpr, label=f"KNN (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig('results/knn_roc_curve.png')
plt.close()

print("Evaluation complete. Results saved in the 'results/' directory.")


import joblib

# Save the pipeline
joblib.dump(knn_pipeline, 'models/knn_pipeline.pkl')
print("Pipeline saved successfully.")