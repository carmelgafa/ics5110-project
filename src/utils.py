'''utility functions'''

import os
from dataclasses import dataclass
import pandas as pd
import torch

@dataclass
class ProjectFolders:
    ''' A dataclass containing the project folders. '''

    RESULTS_FOLDER = "../results"
    DATA_FOLDER = "../data"
    TEMP_FOLDER = "../tmp"

    FINAL_DATASET_FILE =  os.path.join(DATA_FOLDER, "final_dataset.csv")


def get_info(df: pd.DataFrame):
    ''' Provides an overview of a DataFrame's structure and content.'''

    info = df.dtypes.to_frame('dtypes')
    info['non_null'] = df.count()
    info['unique_values'] = df.apply(lambda srs: len(srs.unique()))
    info['first_row'] = df.iloc[0]
    info['last_row'] = df.iloc[-1]
    return info


def compare_score(predictions_val, y_two_year_recid_val_tensor, threshold=0.4):
    ''' Compare the predictions to the actual values '''

    def categorize_score(score):
        ''' Categorize the score based on the threshold '''
        if score <= threshold:
            return 0
        else:   
            return 1

    # Add the predictions to the dataframe by mapping t
    # the categorize_score function to the predictions
    prob_recid_tensor = pd.Categorical(
        pd.Series(predictions_val).map(categorize_score),
    )

    prob_recid_tensor = torch.Tensor(prob_recid_tensor)
    y_two_year_recid_tensor = torch.Tensor( y_two_year_recid_val_tensor)

    # dictionary to store the counts
    counts = {
        "prob_recid": [0, 0, 1, 1],
        "y_two_year_recid_val": [0, 1, 0, 1],
        "count": [0, 0, 0, 0]
    }

    # Count occurrences
    for p, y in zip(prob_recid_tensor, y_two_year_recid_tensor):
        if p == 0 and y == 0:
            counts["count"][0] += 1
        elif p == 0 and y == 1:
            counts["count"][1] += 1
        elif p == 1 and y == 0:
            counts["count"][2] += 1
        elif p == 1 and y == 1:
            counts["count"][3] += 1

    # transform the counts into a dataframe
    df_counts = pd.DataFrame(counts)

    true_neg = df_counts[(df_counts["prob_recid"] == 0) &
        (df_counts["y_two_year_recid_val"] == 0)]['count'].values[0]
    false_pos = df_counts[(df_counts["prob_recid"] == 1) &
        (df_counts["y_two_year_recid_val"] == 0)]['count'].values[0]
    false_neg = df_counts[(df_counts["prob_recid"] == 0) &
        (df_counts["y_two_year_recid_val"] == 1)]['count'].values[0]
    true_pos = df_counts[(df_counts["prob_recid"] == 1) &
        (df_counts["y_two_year_recid_val"] == 1)]['count'].values[0]

    return true_neg, false_pos, false_neg, true_pos


def calculate_metrics(true_neg, false_pos, false_neg, true_pos):
    ''' Calculate the metrics for the confusion matrix '''

    senstivity = true_pos / (true_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    precision = true_pos / (true_pos + false_pos)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    return senstivity, specificity, precision, accuracy
