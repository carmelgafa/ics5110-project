# ICS5110


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


## Execution order feature engineering

Data files are downloaded and saved in the **data** folder. Data preprocessing is carried out in the **utils** folder, as follows:

1. data_aquisition.py
2. data_integrity_check.py
3. feature_engineering.py
4. test_train_split.py

## Execution experiments

Experiments for each technique are stored in the **experiments** folder, as follows:

1. knn_evaluation.py
2. neural_network_evaluation.py
3. logistic_regression_evaluation.py

Each experiment does the following:

1. loads the test and train datasets
2. fits the specified model pipeline on the train data
3. evaluates its performance on the test data using the functions explain previously
4. saves the trained pipeline to a file in the **models** folder

Hyperparameter tuning is carried out in the **experiments** folder, as follows:

1. knn_grid_search.py
2. neural_network_grid_search.py
3. logistic_regression_grid_search.py


## Production

You can try the models in the **production** folder

1. predict_knn.py
2. predict_neural_network.py
3. predict_logistic_regression.py

This folder also has the Gradio app, which can be accessed at http://carmelgafa.github.io/ics5110-project/
