import pandas as pd
from models.logistic_regression import logistic_pipeline


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
logistic_pipeline.fit(X_train, y_train)

from pipeline_evaluation import evaluate_pipeline
evaluate_pipeline('lr', logistic_pipeline, X_test, y_test)



# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # Predict probabilities and classes
# y_pred = logistic_pipeline.predict(X_test)
# y_pred_proba = logistic_pipeline.predict_proba(X_test)[:, 1]



# # save y_pred_preda and y_test in csv

# y_test.to_csv('results/logistic_regression_y_test.csv', index=False)
# pd.DataFrame(y_pred_proba).to_csv('results/logistic_regression_y_pred_proba.csv', index=False)






# # Evaluate metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred_proba)






# # Print metrics
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")
# print(f"ROC-AUC: {roc_auc:.4f}")


# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # Generate confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Display confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic_pipeline.classes_)
# disp.plot()


# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt

# # Compute ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.show()

import joblib

# Save the pipeline
joblib.dump(logistic_pipeline, 'models/logistic_pipeline.pkl')
print("Pipeline saved successfully.")