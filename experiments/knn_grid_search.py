import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from models.knn import create_knn_pipeline
from utils.preprocessing_pipeline import preprocessor

# Load train data
train_data = pd.read_csv('data/train/train_compas-scores-two-years.csv')
X_train = train_data.drop(columns=['two_year_recid'])
y_train = train_data['two_year_recid']

# Define parameter grid for grid search
param_grid = {
    'classifier__n_neighbors': [3, 5, 7],        # Number of neighbors
    'classifier__weights': ['uniform', 'distance'],  # Weighting strategy
    'classifier__metric': ['minkowski'],         # Distance metric
    'classifier__p': [1, 2]                     # Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
}

# Create pipeline
knn_pipeline = create_knn_pipeline()

# Grid search with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=knn_pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)

# Perform grid search
print("Starting grid search...")
grid_search.fit(X_train, y_train)
print("Grid search complete.")

# Save the best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_
with open('results/knn_best_params.txt', 'w') as f:
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"Best Cross-Validation Accuracy: {best_score:.4f}\n")

# Save full results to a CSV file
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('results/knn_grid_search_results.csv', index=False)

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.4f}")
print("Results saved to the 'results/' directory.")
