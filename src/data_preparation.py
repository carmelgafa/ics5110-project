'''prepares the dataset'''

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd


class ColumnReducer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_keep:list)->None:
        self.columns_to_keep = columns_to_keep

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
    def __init__(self, n_neighbors=5, columns=[]):
        self.n_neighbors = n_neighbors
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        for col in self.columns:
            X[col] = imputer.fit_transform(X[[col]])
        return X
    
class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder()
        for col in self.columns:
            X[col] = X[col].astype('category')
            ohe_col = pd.DataFrame(ohe.fit_transform(X[[col]]).toarray(), columns=ohe.get_feature_names_out([col]))
            X = pd.concat([X, ohe_col], axis=1)
            X.drop(col, axis=1, inplace=True)
        return X


class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        for col in self.columns:
            X[col] = scaler.fit_transform(X[[col]])
        return X



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

    standard_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest', 'c_jail_time', 'days_in_custody']

    print(df_compas.describe())


    imputation_transformer = pipeline = Pipeline(steps=[
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
        ('imputer', KNNImputeTransformer(columns=columns_needing_imputation)),
        ('one_hot_encoder', OneHotEncoderTransformer(columns=categorical_columns)),
        ('standard_scaler', StandardScalerTransformer(columns=standard_columns)),
    ])

    # Apply the pipeline
    df_x = pipeline.fit_transform(df_compas)
    
    print(df_x.describe())
