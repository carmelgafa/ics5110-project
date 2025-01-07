'''data preparation'''

import os
import io
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
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
report.add_data_frame(df_compas.head(), 'compas_data')
report.add_data_frame(pd.DataFrame(df_compas.dtypes), 'compas_dtypes')
report.add_data_frame(df_compas.count(), 'compas_count')
report.add_data_frame(df_compas.describe(), 'compas_describe')


# reduce dataset to required fields
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

# calculate the jail time
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
df_reduced = df_reduced.drop(['c_jail_in', 'c_jail_out', 'in_custody', 'out_custody'], axis=1)

import matplotlib.pyplot as plt
# plot histogram of all numerical features

numerical_features = ['age', 'juv_fel_count', 'juv_misd_count',
                      'juv_other_count', 'priors_count', 'days_b_screening_arrest',
                      'days_in_jail', 'days_in_custody', 'decile_score',
                      'two_year_recid']

df_reduced.hist(bins=50, figsize=(20, 15))
plt.savefig(os.path.join(TEMP_FOLDER, 'histogram.png'))
report.add_current_plt(os.path.join(TEMP_FOLDER, 'histogram.png'), 'histogram')


#one hot encode sex, race, c_charge_degree, age_cat
ohe = OneHotEncoder()
ohe_features = ['sex', 'race', 'c_charge_degree', 'age_cat']
df_ohe = pd.DataFrame(
    ohe.fit_transform(df_reduced[ohe_features]).toarray(),
    columns=ohe.get_feature_names_out(ohe_features))
df_reduced = pd.concat([df_reduced, df_ohe], axis=1)

df_reduced = df_reduced.drop(ohe_features, axis=1)

# in days_b_screening_arrest identify the values outside
# the range of -30 to 30 and limit them to the threshold
df_reduced['days_b_screening_arrest'] = df_reduced['days_b_screening_arrest'].clip(-30, 30)

# use knn to impute the missing values in days_b_screening_arrest

imputer = KNNImputer(n_neighbors=5)
df_reduced['days_b_screening_arrest'] = imputer.fit_transform(
    df_reduced[['days_b_screening_arrest']])

# split it into train val and test. keep y into the dataset#
df_reduced_train, df_reduced_val = train_test_split(df_reduced, test_size=0.2, random_state=42)
df_reduced_val, df_reduced_test = train_test_split(df_reduced_val, test_size=0.5, random_state=42)

df_split_results = pd.DataFrame({
    'train': [df_reduced_train.shape[0]],
    'val': [df_reduced_val.shape[0]],
    'test': [df_reduced_test.shape[0]]
})
report.add_data_frame(df_split_results, 'split_results')

# save the datasets
df_reduced.to_csv(os.path.join(DATA_FOLDER, 'df_reduced.csv'), index=False)

df_reduced_train.to_csv( os.path.join(DATA_FOLDER, 'df_reduced_train.csv'), index=False)
df_reduced_val.to_csv(os.path.join(DATA_FOLDER, 'df_reduced_val.csv'), index=False)
df_reduced_test.to_csv(os.path.join(DATA_FOLDER, 'df_reduced_test.csv'), index=False)

report.save()
