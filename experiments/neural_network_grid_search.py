import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from models.neural_network import nn_pipeline
from sklearn.pipeline import Pipeline

# Load train data
train_data = pd.read_csv('data/train/train_compas-scores-two-years.csv')
X_train = train_data.drop(columns=['two_year_recid'])
y_train = train_data['two_year_recid']

# Define parameter grid for grid search
param_grid = {
    'classifier__epochs': [10, 20, 50, 100],          # Number of training epochs
    'classifier__batch_size': [16, 32, 64], # Batch sizes
    'classifier__learning_rate': [0.001, 0.01, 0.1]  # Learning rates
}


# Grid search with StratifiedKFold
cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=nn_pipeline, param_grid=param_grid, cv=cross_val, scoring='accuracy', verbose=1, n_jobs=-1)

# Perform grid search
print("Starting grid search...")
grid_search.fit(X_train, y_train)
print("Grid search complete.")

# Save the best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_
with open('results/nn_best_params.txt', 'w') as f:
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"Best Cross-Validation Accuracy: {best_score:.4f}\n")

# Save full results to a CSV file
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('results/nn_grid_search_results.csv', index=False)

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.4f}")
print("Results saved to the 'results/' directory.")
