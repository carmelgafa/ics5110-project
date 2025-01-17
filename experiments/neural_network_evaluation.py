import pandas as pd
from models.neural_network import nn_pipeline  # Import the neural network pipeline


# # Load train and test data
# train_data = pd.read_csv('data/train/train_compas-scores-two-years.csv')
# test_data = pd.read_csv('data/test/test_compas-scores-two-years.csv')

# # Separate features and target
# X_train = train_data.drop(columns=['two_year_recid'])  # Adjust to your target column
# y_train = train_data['two_year_recid']

# X_test = test_data.drop(columns=['two_year_recid'])
# y_test = test_data['two_year_recid']

from pipeline_load_data import load_pipeline_data
X_train, y_train, X_test, y_test = load_pipeline_data()


# Train the logistic pipeline
nn_pipeline.fit(X_train, y_train)

from pipeline_evaluation import evaluate_pipeline
evaluate_pipeline('nn', nn_pipeline, X_test, y_test)


# # Evaluate the pipeline
# print("Evaluating the neural network pipeline...")
# y_pred = nn_pipeline.predict(X_test)
# y_pred_proba = nn_pipeline.predict_proba(X_test)

# # Metrics
# accuracy = accuracy_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])  # Use the probability for the positive class
# print(f"Test Accuracy: {accuracy:.4f}")
# print(f"ROC AUC: {roc_auc:.4f}")

# # Classification Report
# report = classification_report(y_test, y_pred)
# print("Classification Report:")
# print(report)

# # Save metrics to a file
# with open('results/nn_evaluation_metrics.txt', 'w') as f:
#     f.write(f"Test Accuracy: {accuracy:.4f}\n")
#     f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
#     f.write("Classification Report:\n")
#     f.write(report)

# # Confusion Matrix
# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nn_pipeline.named_steps['classifier'].classes_)
# disp.plot()
# plt.savefig('results/nn_confusion_matrix.png')
# plt.close()



# # ROC Curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
# plt.figure()
# plt.plot(fpr, tpr, label=f"Neural Network (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.savefig('results/nn_roc_curve.png')
# plt.close()

# print("Evaluation complete. Results saved in the 'results/' directory.")


import joblib

# Save the pipeline
joblib.dump(nn_pipeline, 'models/nn_pipeline.pkl')
print("Pipeline saved successfully.")