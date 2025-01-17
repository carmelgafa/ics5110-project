import pandas as pd

def load_pipeline_data():
    
    # Load train and test data
    train_data = pd.read_csv('data/train/train_compas-scores-two-years.csv')
    test_data = pd.read_csv('data/test/test_compas-scores-two-years.csv')

    # Separate features and target
    X_train = train_data.drop(columns=['two_year_recid'])  # Adjust to your target column
    y_train = train_data['two_year_recid']

    X_test = test_data.drop(columns=['two_year_recid'])
    y_test = test_data['two_year_recid']
    
    return X_train, y_train, X_test, y_test