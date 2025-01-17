import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from models.logistic_regression import logistic_pipeline

# Load the training dataset
train_data = pd.read_csv('data/train/train_compas-scores-two-years.csv')
X_train = train_data.drop(columns=['two_year_recid'])  # Adjust target column name as needed
y_train = train_data['two_year_recid']

#Define the parameter grid for logistic regression
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100]  # Regularization strength for Logistic Regression
}


# Define stratified k-fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=param_grid,
    cv=stratified_kfold,
    scoring='accuracy',
    n_jobs=-1  # Use all available processors
)

# Perform grid search
print("Starting grid search...")
grid_search.fit(X_train, y_train)
print("Grid search complete.")

# Save the best parameters
best_params = grid_search.best_params_
with open('results/logistic_regression_best_params.txt', 'w') as f:
    f.write(str(best_params))

# Print results
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Save full results (optional)
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('results/logistic_regression_cv_results.csv', index=False)
