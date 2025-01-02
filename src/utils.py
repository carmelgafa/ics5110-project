'''utility functions'''

import os
from dataclasses import dataclass
import pandas as pd
import torch

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


def compare_score(predictions_val, y_two_year_recid_val_tensor):
        

    PROB_RECID_THRESHOLD = 0.5


    def categorize_score(score):
        if score <= PROB_RECID_THRESHOLD:
            return 0
        else:   
            return 1

    # Add the predictions to the dataframe by mapping the categorize_score function to the predictions
    # prediction values will be low, medium, or high
    prob_recid_tensor = pd.Categorical(
        pd.Series(predictions_val).map(categorize_score),
    )


    preb_recid_tensor = torch.Tensor(prob_recid_tensor)
    y_two_year_recid_tensor = torch.Tensor( y_two_year_recid_val_tensor)

    # Initialize counters
    counts = {
        "prob_recid": [0, 0, 1, 1],
        "y_two_year_recid_val": [0, 1, 0, 1],
        "count": [0, 0, 0, 0]
    }

    # Count occurrences
    for p, y in zip(preb_recid_tensor, y_two_year_recid_tensor):
        if p == 0 and y == 0:
            counts["count"][0] += 1
        elif p == 0 and y == 1:
            counts["count"][1] += 1
        elif p == 1 and y == 0:
            counts["count"][2] += 1
        elif p == 1 and y == 1:
            counts["count"][3] += 1

    # Create DataFrame
    df_counts = pd.DataFrame(counts)

    TN = df_counts[(df_counts["prob_recid"] == 0) & (df_counts["y_two_year_recid_val"] == 0)]['count'].values[0]
    FP = df_counts[(df_counts["prob_recid"] == 1) & (df_counts["y_two_year_recid_val"] == 0)]['count'].values[0]
    FN = df_counts[(df_counts["prob_recid"] == 0) & (df_counts["y_two_year_recid_val"] == 1)]['count'].values[0]
    TP = df_counts[(df_counts["prob_recid"] == 1) & (df_counts["y_two_year_recid_val"] == 1)]['count'].values[0]

    print(f"TN: {TN}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"TP: {TP}")


    Senstivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Precision = TP / (TP + FP)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"Senstivity: {Senstivity}")
    print(f"Specificity: {Specificity}")
    print(f"Precision: {Precision}")
    print(f"Accuracy: {Accuracy}")