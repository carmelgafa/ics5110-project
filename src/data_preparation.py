'''data preparation'''

import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer


df_compas_path = os.path.join('data', 'compas-scores-two-years.csv')
df_compas = pd.read_csv(df_compas_path)

print(df_compas.info())
print(df_compas.head())
print(df_compas.describe())

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

print(df_reduced_train.shape)
print(df_reduced_val.shape)
print(df_reduced_test.shape)

# save the datasets
df_reduced.to_csv('../data/df_reduced.csv', index=False)

df_reduced_train.to_csv('../data/df_reduced_train.csv', index=False)
df_reduced_val.to_csv('../data/df_reduced_val.csv', index=False)
df_reduced_test.to_csv('../data/df_reduced_test.csv', index=False)
