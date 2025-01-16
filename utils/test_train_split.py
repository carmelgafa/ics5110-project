import os
from sklearn.model_selection import train_test_split
import pandas as pd



def split_train_test(data_folder, file_name, train_folder, test_folder, test_size=0.2, random_state=42):
    
    date_file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(date_file_path)
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    

    train_file_path = os.path.join(train_folder, 'train_' + file_name)
    test_file_path = os.path.join(test_folder, 'test_' + file_name)
    
    
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)
    
    
if __name__ == "__main__":
    
    data_folder = 'data/processed'
    file_name = 'df_reduced.csv'
    train_folder = 'data/train'
    test_folder = 'data/test'

    split_train_test(data_folder, file_name, train_folder, test_folder)