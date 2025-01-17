import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch


def evaluate_pipeline(pipeline_name:str,pipeline:torch.nn.Module, X_test, y_test):
    
    # Evaluate the pipeline
    print("Evaluating the neural network pipeline...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])  # Use the probability for the positive class
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Classification Report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # Save metrics to a file
    with open(f'results/{pipeline_name}_evaluation_metrics.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion Matrix
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.named_steps['classifier'].classes_)
    disp.plot()
    plt.savefig(f'results/{pipeline_name}_confusion_matrix.png')
    plt.close()



    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=f"{pipeline_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig('results/{pipeline_name}_roc_curve.png')
    plt.close()

    print("Evaluation complete. Results saved in the 'results/' directory.")
