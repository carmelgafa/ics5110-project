import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
label_encoder = LabelEncoder()
scaler = StandardScaler()
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from offence_mapping import assign_category

import seaborn as sns

df_compas_path = os.path.join('data', 'compas-scores-two-years.csv')
df_compas = pd.read_csv(df_compas_path)

df_reduced = df_compas[['sex', 'two_year_recid', 'decile_score', 'priors_count',
                        'age', 'age_cat', 'race', 'c_charge_degree',
                        'c_charge_desc',
                        'juv_fel_count', 'juv_misd_count', 'juv_other_count']]


# Copy the original dataset
df_encoded = df_reduced.copy()

df_encoded['c_charge_cat'] = df_encoded['c_charge_desc'].apply(assign_category)



# onehot encoding of sex, race, c_charge_degree, age_cat
ohe_features = ['sex', 'race', 'c_charge_degree', 'age_cat', 'c_charge_cat']
ohe = OneHotEncoder()
df_ohe = pd.DataFrame(ohe.fit_transform(df_encoded[ohe_features]).toarray(), columns=ohe.get_feature_names_out(ohe_features))
df_encoded = pd.concat([df_encoded, df_ohe], axis=1)



# final dataset that will serve as starting place for all models
df_final = df_encoded[['two_year_recid', 'decile_score',
                       'priors_count', 
                       'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
                       'sex_Female', 
                       'sex_Male', 
                       'race_African-American','race_Asian','race_Caucasian',
                       'race_Hispanic','race_Native American','race_Other',
                    #    'c_charge_degree_M',
                       'c_charge_degree_F',
                       'age',
                       'age_cat_Greater than 45',
                       'age_cat_Less than 25',
                        # # 'c_charge_cat_Arson',
                        # # 'c_charge_cat_Disorder',
                        # # 'c_charge_cat_Domestic',   
                        # 'c_charge_cat_Fraud',
                        # 'c_charge_cat_Intoxication',
                        # # 'c_charge_cat_Morality',
                        # # 'c_charge_cat_Other',
                        # 'c_charge_cat_Relapse',
                        # # 'c_charge_cat_Theft',
                        # # 'c_charge_cat_Traffic',
                        # 'c_charge_cat_Violence',
                        # 'c_charge_cat_Weapons'
                       ]]

columns_to_scale = ['priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']
for column in columns_to_scale:
    df_final.loc[:, column] = df_final.loc[:, column].astype(float)
    df_final.loc[:, column] = scaler.fit_transform(df_final[[column]])


DECILE_LABEL = 'decile_score'
TWO_YEAR_REC_LABEL = 'two_year_recid'

# label is decile score -- normalize
y = df_final[DECILE_LABEL].values / 10

y_two_year_recid = df_final[TWO_YEAR_REC_LABEL].values

# features
X = df_final.drop(columns=[DECILE_LABEL, TWO_YEAR_REC_LABEL]).values
number_of_features = len(pd.DataFrame(X).columns)
print(f'Number of features: {number_of_features}')

# First, split into train+validation and test datasets
X_train_val, X_test, y_train_val, y_test, y_two_year_recid_train_val, y_two_year_recid_test = train_test_split(
    X, y, y_two_year_recid, test_size=0.2, random_state=42
)

# Then, split train+validation into train and validation datasets
X_train, X_val, y_train, y_val, y_two_year_recid_train, y_two_year_recid_val = train_test_split(
    X_train_val, y_train_val, y_two_year_recid_train_val, test_size=0.25, random_state=42
)
# Note: 0.25 of the train+validation set = 0.2 of the total dataset, resulting in 60/20/20 split
# Convert datasets to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Convert the binary labels (two_year_recid) to tensors
y_two_year_recid_train_tensor = torch.tensor(y_two_year_recid_train, dtype=torch.float32)
y_two_year_recid_val_tensor = torch.tensor(y_two_year_recid_val, dtype=torch.float32)
y_two_year_recid_test_tensor = torch.tensor(y_two_year_recid_test, dtype=torch.float32)

# Print dataset sizes for verification
print(f"Train set: {X_train_tensor.shape}, {y_train_tensor.shape}")
print(f"Validation set: {X_val_tensor.shape}, {y_val_tensor.shape}")
print(f"Test set: {X_test_tensor.shape}, {y_test_tensor.shape}")


# considering the train dataset, decile score varies with age as a box plot
df_train = pd.DataFrame(X_train, columns=df_final.drop(columns=[DECILE_LABEL, TWO_YEAR_REC_LABEL]).columns)
df_train[DECILE_LABEL] = y_train


# seperate age into  3 bins and do a box plot of decile score vs age

# df_train['age_cat'] = pd.cut(df_train['age'], bins=3, labels=['young', 'middle', 'old'])
# sns.boxplot(data=df_train, x='age_cat', y='decile_score')
# plt.title('Decile Score vs Age Category')
# plt.show()


# # separate prior count into 3 bins and do a box plot of decile score vs prior count
# df_train['priors_cat'] = pd.cut(df_train['priors_count'], bins=3, labels=['low', 'medium', 'high'])
# sns.boxplot(data=df_train, x='priors_cat', y='decile_score')
# plt.title('Decile Score vs Priors Count Category')
# plt.show()


# # i want to view tha avarage decile score for each vcombianation of age and prior count
# df_train['age_cat'] = pd.cut(df_train['age'], bins=3, labels=['young', 'middle', 'old'])
# df_train['priors_cat'] = pd.cut(df_train['priors_count'], bins=3, labels=['low', 'medium', 'high'])
# df_train.groupby(['age_cat', 'priors_cat'])['decile_score'].mean().unstack().plot(kind='bar')
# plt.title('Decile Score vs Age and Priors Count')
# plt.show()

# # i want to superimpose two years recidivism (that is in y_two_year_recid) on the above plot
# df_train['two_year_recid'] = y_two_year_recid_train
# df_train.groupby(['age_cat', 'priors_cat'])['two_year_recid'].mean().unstack().plot(kind='bar')
# plt.title('Two Year Recidivism vs Age and Priors Count')
# plt.show()


# i want to a table with tha avarage decile score for each vcombianation of age and prior count
df_train['age_cat'] = pd.cut(df_train['age'], bins=3, labels=['young', 'middle', 'old'])
df_train['priors_cat'] = pd.cut(df_train['priors_count'], bins=3, labels=['low', 'medium', 'high'])
df_table_decile = df_train.groupby(['age_cat', 'priors_cat'])['decile_score'].mean().unstack()
print(df_table_decile)

# repeat, this time however with the percentage of recidivism
df_train['two_year_recid'] = y_two_year_recid_train
df_table_recid = df_train.groupby(['age_cat', 'priors_cat'])['two_year_recid'].mean().unstack()
print(df_table_recid)

#multiply them
df_table_combined = df_table_decile * df_table_recid
print(df_table_combined)




# sns.boxplot(data=df_train, x='age', y='decile_score')
# plt.title('Decile Score vs Age')
# plt.show()

# sns.boxplot(data=df_train, x='priors_count', y='decile_score')
# plt.title('Decile Score vs Priors Count')
# plt.show()


# for the trainset, list the percentage recid for each decile score
df_train['decile_score'] = y_train
df_train['two_year_recid'] = y_two_year_recid_train
df_table = df_train.groupby('decile_score')['two_year_recid'].mean()
print(df_table)

# call compare_score with values between 0.3 and 0.6 in steps of 0.5 to and list the values of sensitivity, specificity, precision, and accuracy for each value
from utils import compare_score

for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    Senstivity, Specificity, Precision, Accuracy = compare_score(y_val, y_two_year_recid_val_tensor, threshold=threshold)
    print(f"Threshold: {threshold}", f"Senstivity: {Senstivity}", f"Specificity: {Specificity}", f"Precision: {Precision}", f"Accuracy: {Accuracy}")
