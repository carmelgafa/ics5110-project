import pandas as pd

y_test = pd.read_csv('results/logistic_regression_y_test.csv')
y_pred_proba = pd.read_csv('results/logistic_regression_y_pred_proba.csv')


y_test = y_test.squeeze()
y_pred_proba = y_pred_proba.squeeze()

y_pred = (y_pred_proba > 0.5) * 1.0

#create dataframe of y_test and y_pred
y_comparison = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})


true_positives_sum = y_comparison[(y_comparison['y_test'] == 1) & (y_comparison['y_pred'] == 1)].shape[0]
false_positives_sum = y_comparison[(y_comparison['y_test'] == 0) & (y_comparison['y_pred'] == 1)].shape[0]
true_negatives_sum = y_comparison[(y_comparison['y_test'] == 0) & (y_comparison['y_pred'] == 0)].shape[0]
false_negatives_sum = y_comparison[(y_comparison['y_test'] == 1) & (y_comparison['y_pred'] == 0)].shape[0]

accuracy = (true_positives_sum + true_negatives_sum) / len(y_comparison)
precision = true_positives_sum / (true_positives_sum + false_positives_sum)
recall = true_positives_sum / (true_positives_sum + false_negatives_sum)
f1 = 2 * (precision * recall) / (precision + recall)

# Equation results
# Accuracy: 0.6909
# Precision: 0.6605
# Recall: 0.5774
# F1: 0.6162

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")


import numpy as np

true_positive_rates = []
false_positive_rates = []
area_under_curve = 0

for threshold in np.arange(0, 1, 0.01):
    y_pred = (y_pred_proba > threshold) * 1.0
    y_comparison = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    
    true_positives_sum = y_comparison[(y_comparison['y_test'] == 1) & (y_comparison['y_pred'] == 1)].shape[0]
    false_positives_sum = y_comparison[(y_comparison['y_test'] == 0) & (y_comparison['y_pred'] == 1)].shape[0]
    true_negatives_sum = y_comparison[(y_comparison['y_test'] == 0) & (y_comparison['y_pred'] == 0)].shape[0]
    false_negatives_sum = y_comparison[(y_comparison['y_test'] == 1) & (y_comparison['y_pred'] == 0)].shape[0]
    
    accuracy = (true_positives_sum + true_negatives_sum) / len(y_comparison)
    precision = true_positives_sum / (true_positives_sum + false_positives_sum)
    
    true_positive_rates.append(true_positives_sum / (true_positives_sum + false_negatives_sum))
    false_positive_rates.append(false_positives_sum / (false_positives_sum + true_negatives_sum))



# reverse order of false_positive_rates and true_positive_rates
true_positive_rates = true_positive_rates[::-1]
false_positive_rates = false_positive_rates[::-1]

for i in range(1, len(true_positive_rates)):
    area_under_curve += (true_positive_rates[i] + true_positive_rates[i - 1]) * (false_positive_rates[i] - false_positive_rates[i - 1]) / 2
    


# Plot ROC Curve
import matplotlib.pyplot as plt

plt.figure()
lw = 2
plt.plot(false_positive_rates, true_positive_rates, color='darkorange',
         lw=lw, label='Logistic Regression (area = %0.2f)')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(f"Area Under the Curve: {area_under_curve:.4f}")






