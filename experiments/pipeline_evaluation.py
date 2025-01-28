import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch


def evaluate_pipeline(pipeline_name:str,pipeline:torch.nn.Module, X_test, y_test):
    '''
    Evaluate a machine learning pipeline on the test data.
    '''

    # Evaluate the pipeline
    print(f"Evaluating the {pipeline_name} pipeline...")

    # predict() is used to predict the actual class (in your case one of 0, 1).
    # predict_proba() is used to predict the class probabilities,
    # that is two values for each prediction
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    print(pipeline.named_steps['classifier'].classes_)

    create_evaluation_report(pipeline_name, y_test, y_pred, y_pred_proba)

def create_evaluation_report(pipeline_name:str, y_test, y_pred, y_pred_proba):
    '''
    Evaluate a machine learning pipeline on the test data.
    The following metrics are calculated:
    - Test Accuracy
    - ROC AUC
    - Classification Report
    - Confusion Matrix
    - ROC Curve
    '''

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1]) 
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Classification Report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # Save metrics to a file
    with open(f'results/{pipeline_name}_evaluation_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['0', '1'])
    cm_display.plot()
    plt.savefig(f'results/{pipeline_name}_confusion_matrix.png')
    plt.show()


    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=f"{pipeline_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig('results/{pipeline_name}_roc_curve.png')
    plt.show()

    print("Evaluation complete. Results saved in the 'results/' directory.")
    
    TP = confusion_matrix(y_test, y_pred)[1, 1]
    TN = confusion_matrix(y_test, y_pred)[0, 0]
    FP = confusion_matrix(y_test, y_pred)[0, 1]
    FN = confusion_matrix(y_test, y_pred)[1, 0]
    
    
    
    f1_class_0 = 2 * TN / (2 * TN + FP + FN)
    f1_class_1 = 2 * TP / (2 * TP + FP + FN)
    precision_class_1 = TP / (TP + FP)
    overall_accuracy = (TP + TN) / (TP + TN + FP + FN)
    weighted_average_f1 = (f1_class_0 + f1_class_1) / 2

    create_document_line(pipeline_name, 
        f1_class_0,
        f1_class_1,
        precision_class_1,
        overall_accuracy,
        weighted_average_f1)



def create_document_line(pipeline_name:str,
        f1_class_0:float,
        f1_class_1:float,
        precision_class_1:float,
        overall_accuracy:float,
        weighted_average_f1:float):
    '''
    Creates a latex table line to display most important evaluation metrics
    
    Technique Name & 
    \\makecell{F1 class 0 \\ F1 class 1}  & 
    Precision class 1 &
    OverallAccuracy &
    Weighted Average F1\\ \\hline
    '''
    print(f"{pipeline_name} & \\makecell {{ {f1_class_0:.2f} \\\\ {f1_class_1:.2f} }} & {precision_class_1:.2f} & {overall_accuracy:.2f} & {weighted_average_f1:.2f} \\\\ \\hline")
