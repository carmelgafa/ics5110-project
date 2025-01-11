'''prepares the dataset'''

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd


<<<<<<< HEAD
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
B N

=======
class ColumnReducer(BaseEstimator, TransformerMixin):
    '''reduces the dataset to the necessary columns'''
    def __init__(self, columns_to_keep:list):
        self.columns_to_keep = columns_to_keep
>>>>>>> 5225dd03b07ded7f4db580af221df49978c62ab7

    def fit(self, df_x:pd.DataFrame, y=None)->'ColumnReducer':
        '''no fitting required'''
        return self

    def transform(self, df_x:pd.DataFrame)->pd.DataFrame:
        '''select columns to keep'''
        # Ensure X is a DataFrame to support column selection
        return df_x[self.columns_to_keep]

class JailTimeTransformer(BaseEstimator, TransformerMixin):
    '''calculate jail time'''
    def __init__(self):
        pass

    def fit(self, df_x:pd.DataFrame, y=None):
        '''no fitting required'''
        return self

    def transform(self, df_x:pd.DataFrame)->pd.DataFrame:
        '''calculate jail time'''
        df_x = df_x.copy()
        df_x['c_jail_in'] = pd.to_datetime(
            df_x['c_jail_in'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df_x['c_jail_out'] = pd.to_datetime(
            df_x['c_jail_out'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df_x['c_jail_time'] = abs((df_x['c_jail_out'] - df_x['c_jail_in']).dt.days)
        return df_x.drop(['c_jail_in', 'c_jail_out'], axis=1)

class CustodyTimeTransformer(BaseEstimator, TransformerMixin):
    '''calculate custody time'''
    def __init__(self):
        pass

    def fit(self, df_x:pd.DataFrame, y=None):
        '''no fitting required'''
        return self

    def transform(self, df_x:pd.DataFrame)->pd.DataFrame:
        '''calculate custody time'''
        df_x = df_x.copy()
        df_x['in_custody'] = pd.to_datetime(
            df_x['in_custody'], format='%Y-%m-%d', errors='coerce')
        df_x['out_custody'] = pd.to_datetime(
            df_x['out_custody'], format='%Y-%m-%d', errors='coerce')
        df_x['days_in_custody'] = abs((df_x['out_custody'] - df_x['in_custody']).dt.days)
        return df_x.drop(['in_custody', 'out_custody'], axis=1)

class EndStartTransformer(BaseEstimator, TransformerMixin):
    '''calculate end-start time'''
    def __init__(self):
        pass

    def fit(self, df_x:pd.DataFrame, y=None):
        '''no fitting required'''
        return self

    def transform(self, df_x:pd.DataFrame)->pd.DataFrame:
        '''calculate end-start time'''
        df_x = df_x.copy()
        df_x['end_start'] = abs((df_x['end'] - df_x['start']))
        return df_x


class KNNImputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        return imputed  

class ToDataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert NumPy array to DataFrame
        return pd.DataFrame(X, columns=self.columns)

if __name__ == '__main__':

    import os
    import pandas as pd
    from sklearn.impute import KNNImputer

    DATA_FOLDER = 'data'
    df_compas = pd.read_csv(os.path.join(DATA_FOLDER, 'compas-scores-two-years.csv'))

    # Define the columns to keep
    columns_to_keep = [ "sex", "age", "race", "juv_fel_count", "decile_score",
                   "juv_misd_count", "juv_other_count", "priors_count",
                   "days_b_screening_arrest", "c_jail_in", "c_jail_out",
                   "c_days_from_compas", "c_charge_degree", "v_decile_score",
                   "in_custody", "out_custody", "start", "end", "event", "two_year_recid" ]


    columns_needing_imputation = ['days_b_screening_arrest', 'c_jail_time', 'days_in_custody']
    
    categorical_columns = ['race', 'c_charge_degree', 'sex']

    imputation_transformer = pipeline = Pipeline(steps=[
        # ('jail_time', JailTimeTransformer()),
        # ('custody_time', CustodyTimeTransformer()),
        # ('end_start', EndStartTransformer()),
        ('imputation_pipeline', KNNImputeTransformer(n_neighbors=5)),
    ])
    
    from sklearn.compose import ColumnTransformer
    processor = ColumnTransformer(
        transformers=[
            ('imputation', imputation_transformer, columns_needing_imputation)
        ]
    )


    pipeline = Pipeline(steps=[
        ('column_selector', ColumnReducer(columns_to_keep=columns_to_keep)),
        ('jail_time', JailTimeTransformer()),
        ('custody_time', CustodyTimeTransformer()),
        ('end_start', EndStartTransformer()),
        ('imputation_pipeline', processor),
        ('to_dataframe', ToDataFrameTransformer(columns=columns_needing_imputation))

    ])

    # Apply the pipeline
    df_x = pipeline.fit_transform(df_compas)

<<<<<<< HEAD

# split it into train val and test. keep y into the dataset#
# ---------------------------------------------------------

stratify_column = 'race'

# Set up StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Create train and test sets
for train_idx, test_idx in split.split(df_reduced, df_reduced[stratify_column]):
    df_train = df_reduced.iloc[train_idx]
    df_test = df_reduced.iloc[test_idx]

# # Further split test into dev and test (50/50 split)
# split_dev = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

# for test_idx, val_idx in split_dev.split(df_test, df_test[stratify_column]):
#     df_test_final = df_test.iloc[test_idx]
#     df_val = df_test.iloc[val_idx]

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
    # 'val': [df_val.shape[0]],
    # 'val race percentages': get_race_distribution(df_val),
    'test': [df_test.shape[0]],
    'test race percentages': get_race_distribution(df_test)
})
report.add_data_frame(df_split_results, 'split_results')

# save the datasets
df_reduced.to_csv(os.path.join(DATA_FOLDER, 'df_reduced.csv'), index=False)

df_train.to_csv( os.path.join(DATA_FOLDER, 'train_dataset.csv'), index=False)
# df_val.to_csv(os.path.join(DATA_FOLDER, 'test_dataset.csv'), index=False)
df_test.to_csv(os.path.join(DATA_FOLDER, 'df_reduced_test.csv'), index=False)

report.save()
=======
    print(df_x.info())
>>>>>>> 5225dd03b07ded7f4db580af221df49978c62ab7
