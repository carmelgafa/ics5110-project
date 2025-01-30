'''
data preparation

The following preparation steps are performed:

1. reduce dataset to required fields
2. calculate the jail time and custody time
3. impute the missing values in days_b_screening_arrest
4. encode categorical features
5. split it into train val and test

'''

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import KNNImputer
from report_writer import ReportWriter

DATA_FOLDER = 'data'
RESULTS_FOLDER = 'results'
TEMP_FOLDER = 'tmp'

# start report to hold results
report = ReportWriter(os.path.join(RESULTS_FOLDER, 'data_preparation_report.xlsx'))

# load dataset
df_compas_path = os.path.join(DATA_FOLDER, 'compas-scores-two-years.csv')
df_compas = pd.read_csv(df_compas_path)

# gather the trivial  dataset information
report.add_data_frame(
    pd.DataFrame({'records': [df_compas.shape[0]], 'columns': [df_compas.shape[1]]}),
    'compas_summary')


# reduce dataset to required fields
# ---------------------------------
df_reduced = df_compas[[
    'sex',
    'age',
    'age_cat',
    'race',
    'juv_fel_count',
    'juv_misd_count',
    'juv_other_count',
    'priors_count',
    'days_b_screening_arrest',
    'c_jail_in',
    'c_jail_out',
    'c_charge_degree',
    'c_charge_desc',
    'in_custody',
    'out_custody',
    'decile_score',
    'two_year_recid'
]]

# start data manipulation
df_reduced = df_reduced.copy()

# calculate the jail time and custody time
# ----------------------------------------
df_reduced['c_jail_in'] = pd.to_datetime(
    df_reduced['c_jail_in'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_reduced['c_jail_out'] = pd.to_datetime(
    df_reduced['c_jail_out'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_reduced['days_in_jail'] = abs((df_reduced['c_jail_out'] - df_reduced['c_jail_in']).dt.days)
# not available values should be 0
df_reduced['days_in_jail'] = df_reduced['days_in_jail'].fillna(0)
# ensure that it is an integer
df_reduced['days_in_jail'] = df_reduced['days_in_jail'].astype(int)

# calculate custody time
df_reduced['in_custody'] = pd.to_datetime(
    df_reduced['in_custody'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_reduced['out_custody'] = pd.to_datetime(
    df_reduced['out_custody'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

df_reduced['days_in_custody'] = abs((df_reduced['out_custody'] - df_reduced['in_custody']).dt.days)
df_reduced['days_in_custody'] = df_reduced['days_in_custody'].fillna(0)
df_reduced['days_in_custody'] = df_reduced['days_in_custody'].astype(int)

# remove the days in and days out from the dataset
df_reduced = df_reduced.drop(['c_jail_in', 'c_jail_out',
                              'in_custody', 'out_custody', 
                              'days_in_custody'], axis=1)

report.add_data_frame(df_reduced.head(), 'compas_data')
report.add_data_frame(pd.DataFrame(df_reduced.dtypes), 'compas_dtypes')
report.add_data_frame(df_reduced.count(), 'compas_count')
report.add_data_frame(df_reduced.describe(), 'compas_describe')

# plot histogram of all numerical features

numerical_features = ['age', 'juv_fel_count', 'juv_misd_count',
                      'juv_other_count', 'priors_count', 'days_b_screening_arrest',
                      'days_in_jail', 'decile_score',
                      'two_year_recid']

df_reduced.hist(bins=50, figsize=(20, 15))
plt.savefig(os.path.join(TEMP_FOLDER, 'histogram.png'))
report.add_current_plt(os.path.join(TEMP_FOLDER, 'histogram.png'), 'histogram')


# impute the missing values in days_b_screening_arrest
# ----------------------------------------------------
df_reduced_copy = df_reduced.copy()

# Columns to use as predictors for KNN Imputation (numeric only)
predictor_columns = ['age', 'juv_fel_count', 'juv_misd_count',
                      'juv_other_count', 'priors_count',
                      'days_in_jail', 'decile_score',
                      'two_year_recid']
# target column to fill
TARGET_COLUMN = "days_b_screening_arrest"

# Combine the predictors and target column into a separate DataFrame
df_subset = df_reduced[predictor_columns + [TARGET_COLUMN]]

# KNN imputer to subset
imputer = KNNImputer(n_neighbors=5)
df_subset_imputed = pd.DataFrame(imputer.fit_transform(df_subset), columns=df_subset.columns)

# replace only the target column in the original DataFrame
df_reduced[TARGET_COLUMN] = df_subset_imputed[TARGET_COLUMN]

# Plot distributions before and after imputation
plt.figure(figsize=(12, 6))
sns.kdeplot(df_reduced[TARGET_COLUMN], label="After Imputation", color="green", fill=True)
sns.kdeplot(df_reduced_copy[TARGET_COLUMN], label="Before Imputation", color="red", fill=True)
plt.title("Effect of KNN Imputation on 'days_b_screening_arrest'")
plt.xlabel("Days Between Arrest and Screening")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(TEMP_FOLDER, 'imputation.png'))
report.add_current_plt(os.path.join(TEMP_FOLDER, 'imputation.png'), 'imputation')



# encode categorical features
# ---------------------------
#one hot encode sex, race, c_charge_degree, age_cat
ohe = OneHotEncoder()
ohe_features = ['sex', 'race', 'c_charge_degree']
df_ohe = pd.DataFrame(
    ohe.fit_transform(df_reduced[ohe_features]).toarray(),
    columns=ohe.get_feature_names_out(ohe_features))
df_reduced = pd.concat([df_reduced, df_ohe], axis=1)


# ordinal encoding
oe_features = ['age_cat']
age_cat_order = ['Less than 25', '25 - 45', 'Greater than 45']

ordinal_encoder = OrdinalEncoder(categories=[age_cat_order])
df_reduced[['age_cat_encoded']] = ordinal_encoder.fit_transform(
    df_reduced[['age_cat']]
)

report.add_data_frame(pd.DataFrame(df_reduced.dtypes), 'final_dtypes')


# split it into train val and test. keep y into the dataset#
# ---------------------------------------------------------

stratify_column = 'race'

# Set up StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Create train and test sets
for train_idx, test_idx in split.split(df_reduced, df_reduced[stratify_column]):
    df_train = df_reduced.iloc[train_idx]
    df_test = df_reduced.iloc[test_idx]

# Further split test into dev and test (50/50 split)
split_dev = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for test_idx, val_idx in split_dev.split(df_test, df_test[stratify_column]):
    df_test_final = df_test.iloc[test_idx]
    df_val = df_test.iloc[val_idx]

# Function to print race counts and percentages
def get_race_distribution(df:pd.DataFrame):
    ''' Returns a dataframe with race percentages'''
    race_counts = df['race'].value_counts()
    race_percentages = (race_counts / len(df)) * 100
    result = str(race_percentages.to_dict())
    return result

df_split_results = pd.DataFrame({
    'train': [df_train.shape[0]],
    'train race percentages': get_race_distribution(df_train),
    'val': [df_val.shape[0]],
    'val race percentages': get_race_distribution(df_val),
    'test': [df_test.shape[0]],
    'test race percentages': get_race_distribution(df_test)
})
report.add_data_frame(df_split_results, 'split_results')

# save the datasets
df_reduced.to_csv(os.path.join(DATA_FOLDER, 'df_reduced.csv'), index=False)

df_train.to_csv( os.path.join(DATA_FOLDER, 'train_dataset.csv'), index=False)
df_val.to_csv(os.path.join(DATA_FOLDER, 'test_dataset.csv'), index=False)
df_test.to_csv(os.path.join(DATA_FOLDER, 'df_reduced_test.csv'), index=False)

report.save()
