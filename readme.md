# ICS5510


## Project structure

```text
├── data/
│   ├── raw/ (Original dataset)
│   ├── processed/ (Cleaned dataset)
├── notebooks/
├── models/
├── utils/
│   ├── data_acquisition.py (Script for downloading dataset)
│   ├── data_integrity_check.py (Script for validating data structure and integrity)
|   |── feature_engineering.py (Script for feature engineering pipeline)
|   |── preprocessing_pipeline.py (Script for data preprocessing pipeline)
|   |── test_train_split.py (Script for splitting data into train and test sets)
├── experiments/
│   ├── knn_evaluation.py (Script for evaluating KNN model)
│   ├── neural_network_evaluation.py (Script for evaluating Neural Network model)
│   ├── logistic_regression_evaluation.py (Script for evaluating Logistic Regression model)
│   ├── pipeline_evaluation.py (Script for evaluating the entire pipeline)
│   ├── knn_grid_search.py (Script for performing grid search for KNN model)
│   ├── neural_network_grid_search.py (Script for performing grid search for Neural Network model)
│   ├── logistic_regression_grid_search.py (Script for performing grid search for Logistic Regression model)
├── production/
│   ├── predict_knn.py (Script for predicting using KNN model)
│   ├── predict_neural_network.py (Script for predicting using Neural Network model)
│   ├── predict_logistic_regression.py (Script for predicting using Logistic Regression model)
├── web_app/
├── results/
├── main_pipeline.py
├── README.md - this file
├── requirements.txt
```


## Execution order

1. data_aquisition.py
2. data_integrity_check.py
3. feature_engineering.py
4. test_train_split.py
