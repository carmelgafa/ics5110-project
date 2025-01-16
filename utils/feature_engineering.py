'''preprocessing for compas data'''

import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

# Define the custom transformation
def feature_engineering(df):
    '''
    preprocessing for compas data
    1. keep only cols that make sense
    2. difference between two c_jail_in and c_jail_out -> days_in_jail
    3. difference between two in_custody and out_custody -> days_in_custody
    '''

    # keep only relevant cols
    columns_to_keep = [ "sex", "age", "race", "juv_fel_count", "decile_score",
                "juv_misd_count", "juv_other_count", "priors_count",
                "days_b_screening_arrest", "c_jail_in", "c_jail_out",
                "c_days_from_compas", "c_charge_degree",
                "in_custody", "out_custody", "event", "two_year_recid" ]

    df_reduced = df[columns_to_keep].copy()

    # difference between two c_jail_in and c_jail_out -> days_in_jail
    df_reduced['c_jail_in'] = pd.to_datetime(
        df_reduced['c_jail_in'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_reduced['c_jail_out'] = pd.to_datetime(
        df_reduced['c_jail_out'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_reduced['days_in_jail'] = abs(
        (df_reduced['c_jail_out'] - df_reduced['c_jail_in']).dt.days)

    remove_columns = ['c_jail_in', 'c_jail_out']
    df_reduced = df_reduced.drop(remove_columns, axis=1)

    # difference between two in_custody and out_custody -> days_in_custody
    df_reduced['in_custody'] = pd.to_datetime(
        df_reduced['in_custody'], format='%Y-%m-%d', errors='coerce')
    df_reduced['out_custody'] = pd.to_datetime(
        df_reduced['out_custody'], format='%Y-%m-%d', errors='coerce')
    df_reduced['days_in_custody'] = abs(
        (df_reduced['out_custody'] - df_reduced['in_custody']).dt.days)

    remove_columns = ['in_custody', 'out_custody']
    df_reduced = df_reduced.drop(remove_columns, axis=1)

    return df_reduced

# Wrap the function in a FunctionTransformer
feature_engineering_transformer = FunctionTransformer(feature_engineering, validate=False)


def feature_engineer_data(data_folder, file_name, save_folder, save_name):
    '''preprocess of dataset and save'''

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    data_file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(data_file_path)

    df_reduced = feature_engineering_transformer.transform(df)

    print(df_reduced.info())

    save_file_path = os.path.join(save_folder, save_name)
    df_reduced.to_csv(save_file_path, index=False)


if __name__ == '__main__':
    data_folder = 'data/raw'
    file_name = 'compas-scores-two-years.csv'
    save_folder = 'data/processed'
    save_name = 'df_reduced.csv'
    feature_engineer_data(data_folder, file_name, save_folder, save_name)
