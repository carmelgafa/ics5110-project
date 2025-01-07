'''download the dataset'''

import os

FILE_URL ='https://raw.githubusercontent.com/propublica/compas-analysis/refs/heads/master/compas-scores-two-years.csv'
FILE_NAME = 'compas-scores-two-years.csv'
DATA_FOLDER = 'data'

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

if not os.path.exists(os.path.join(DATA_FOLDER, FILE_NAME)):
    import urllib.request
    urllib.request.urlretrieve(FILE_URL, os.path.join(DATA_FOLDER, FILE_NAME))
