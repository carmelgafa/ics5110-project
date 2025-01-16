'''initial data exploration'''

import os
import pandas as pd
from pandas_profiling import ProfileReport



def initial_data_exploration(data_folder, file_name):

    # Load dataset
    df_compas = pd.read_csv(os.path.join(data_folder, file_name))

    # Preview the data
    print(df_compas.head())

    print(f"Dataset Shape: {df_compas.shape}")
    print(df_compas.info())

    # check for missing values
    missing_data = df_compas.isnull().sum()
    print(f'missing values: {missing_data[missing_data > 0]}')

    # validate column names
    df_compas.columns = df_compas.columns.str.strip().str.lower().str.replace(' ', '_')

    # identify duplicate rows
    duplicate_rows = df_compas[df_compas.duplicated()]
    print(f"Duplicate Rows: {duplicate_rows.shape[0]}")

    # examine statistical summary
    print(df_compas.describe())

    categorical_columns = ['sex', 'race', 'c_charge_degree', 'age_cat']
    print(df_compas[categorical_columns].nunique())
    print([df_compas[categorical_column].unique() for categorical_column in categorical_columns ] )

    # create report
    profile = ProfileReport(df_compas, title="Profiling Report")
    profile.to_file("results/report.html")


if __name__ == "__main__":

    FILE_NAME = 'compas-scores-two-years.csv'
    DATA_FOLDER = 'data/raw'

    initial_data_exploration(DATA_FOLDER, FILE_NAME)
