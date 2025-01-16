'''download the dataset'''
import os

def download_dataset(url:str, file_name:str, data_folder:str)->None:
    '''download the dataset and saves it into a data folder'''

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if not os.path.exists(os.path.join(data_folder, file_name)):
        import urllib.request
        urllib.request.urlretrieve(url, os.path.join(data_folder, file_name))


if __name__ == '__main__':
    FILE_URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/refs/heads/master/compas-scores-two-years.csv'
    FILE_NAME = 'compas-scores-two-years.csv'
    DATA_FOLDER = 'data/raw'


    download_dataset(FILE_URL, FILE_NAME, DATA_FOLDER)
