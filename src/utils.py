'''utility functions'''

import os
from dataclasses import dataclass
import pandas as pd


@dataclass
class ProjectFolders:
    '''
    A dataclass containing the project folders.
    '''

    RESULTS_FOLDER = "../results"
    DATA_FOLDER = "../data"
    TEMP_FOLDER = "../tmp"

    FINAL_DATASET_FILE =  os.path.join(DATA_FOLDER, "final_dataset.csv")



def get_info(df: pd.DataFrame):
    '''
    Provides an overview of a DataFrame's structure and content.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following information about the original DataFrame:
        - dtypes: The data type of each column.
        - non_null: The number of non-null values in each column.
        - unique_values: The number of unique values in each column.
        - first_row: The first row of the original DataFrame.
        - last_row: The last row of the original DataFrame.
    '''

    info = df.dtypes.to_frame('dtypes')
    info['non_null'] = df.count()
    info['unique_values'] = df.apply(lambda srs: len(srs.unique()))
    info['first_row'] = df.iloc[0]
    info['last_row'] = df.iloc[-1]
    return info
