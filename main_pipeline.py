from utils import data_aquisition
from utils import data_integrity_check
from utils import feature_engineering
from utils import test_train_split

def download_analysis_pipeline(file_url, file_name, data_folder):

    # download the dataset
    data_aquisition.download_dataset(file_url, file_name, data_folder)

    # initial data exploration
    data_integrity_check.initial_data_exploration(data_folder, file_name)


def preparation_pipeline(data_folder, file_name, processed_folder, processed_file_name, train_folder, test_folder):
    
    feature_engineering.feature_engineer_data(data_folder, file_name, processed_folder, processed_file_name)

    test_train_split.split_train_test(data_folder, file_name, train_folder, test_folder)


if __name__ == "__main__":
    
    FILE_URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/refs/heads/master/compas-scores-two-years.csv'
    FILE_NAME = 'compas-scores-two-years.csv'
    DATA_FOLDER = 'data/raw'

    # download the dataset and initial data exploration pipeline
    # download_analysis_pipeline(FILE_URL, FILE_NAME, DATA_FOLDER)
    
    PROCESSED_FOLDER = 'data/processed'
    PROCESSED_FILE_NAME = 'df_reduced.csv'
    TRAIN_FOLDER = 'data/train'
    TEST_FOLDER = 'data/test'

    # feature engineering pipeline
    preparation_pipeline(DATA_FOLDER, FILE_NAME, PROCESSED_FOLDER, PROCESSED_FILE_NAME, TRAIN_FOLDER, TEST_FOLDER)
