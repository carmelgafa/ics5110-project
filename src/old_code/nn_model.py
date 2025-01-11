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



# separate age in 3 buckets
age_cat = pd.cut(df_final['age'], bins=[0, 25, 45, 100], labels=['young', 'middle', 'old'])
# do same with prior count in 3 buckets
priors_cat = pd.cut(df_final['priors_count'], bins=[0, 5, 10, 100], labels=['low', 'medium', 'high'])

# create a table with the average decile score of the different groups so we will have the average decile score of (young and low), (young and medium), (young and high), (middle and low), etc.
df_final['age_cat'] = age_cat
df_final['priors_cat'] = priors_cat



# create a table with the average decile score of the different groups so we will have the average decile score of (young and low), (young and medium), (young and high), (middle and low), etc.

# create a table with the average decile score of the different groups so we will have the average decile score of (young and low), (young and medium), (young and high), (middle and low), etc.
df_grouped = df_final.groupby(['age_cat', 'priors_cat']).agg({'decile_score': 'mean'}).reset_index()

print(df_grouped)








def train_model(batch_size, initial_learning_rate, early_stopping_threshold, middle_layer_size=None):

    max_epochs = 3000

    # neural network definition
    layers = [
        nn.Linear(number_of_features, middle_layer_size), nn.ELU(),
        # nn.Linear(middle_layer_size, middle_layer_size), nn.ELU(),
        # nn.Linear(middle_layer_size, middle_layer_size), nn.ELU(),
        # nn.Linear(middle_layer_size, middle_layer_size), nn.ELU(),
        # nn.Linear(middle_layer_size, middle_layer_size), nn.ELU(),
        nn.Linear(middle_layer_size, 1)
        ]

    parameters = [p for layer in layers for p in layer.parameters()]
    number_of_parameters = sum(p.nelement() for p in parameters)
    print(f"Number of parameters: {number_of_parameters}")

    for p in parameters:
        p.requires_grad = True

    training_samples = X_train_tensor.size(0)
    batches_per_epoch = training_samples / batch_size

    training_losses = []
    validation_losses = []

    for epoch in range(max_epochs):

        training_loss = 0.0
        # correct = 0
        total = 0

        for i in range(0, training_samples, batch_size):
            # Mini-batch preparation
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]

            # Forward pass
            x = X_batch
            for layer in layers:
                x = layer(x)

            # Loss calculation
            loss = F.mse_loss(x.squeeze(), y_batch)

            # Clear gradients -- ensure no accumulation of gradients
            for p in parameters:
                if p.grad is not None:
                    p.grad.zero_()

            # Backward pass -- increase learning rate
            loss.backward()
            for p in parameters:
                if epoch < 100:
                    p.data.add_(-initial_learning_rate, p.grad)
                elif epoch < 200:
                    p.data.add_(-initial_learning_rate/10, p.grad)
                elif epoch < 300:
                    p.data.add_(-initial_learning_rate/100, p.grad)
                else:
                    p.data.add_(-initial_learning_rate/1000, p.grad)

            # Track epoch loss
            training_loss += loss.item()
            training_losses.append(loss.item())

        validation_loss_delta = 1

        # validation loss
        with torch.no_grad():
            x = X_val_tensor
            for layer in layers:
                x = layer(x)
            val_loss = F.mse_loss(x.squeeze(), y_val_tensor)
            validation_losses.append(val_loss.item())
            validation_loss_delta = abs(validation_losses[-1] - validation_losses[-2]) if len(validation_losses) > 1 else 0

        # Early stopping
        if epoch > 5 and  validation_loss_delta < early_stopping_threshold:
            print(f"Early stopping at epoch {epoch+1}, {validation_losses[-1]}, {validation_losses[-2]}  delta: {validation_loss_delta}")
            break

        print(f"Epoch [{epoch+1}/{max_epochs}], Training loss: {training_loss/batches_per_epoch:.4f}", f"Validation Loss: {val_loss}, validation_delta: {validation_loss_delta}")

    plt.plot(training_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    plt.plot(validation_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.show()

    return layers


# layers = train_model(batch_size=32, initial_learning_rate=0.1, early_stopping_threshold=1e-8, middle_layer_size=32)
# layers = train_model(batch_size=32, initial_learning_rate=0.05, early_stopping_threshold=0.0001, middle_layer_size=32)


# batch_sizes = [32, 64]
# learning_rates = [0.1, 0.01, 0.001]
# early_stopping_thresholds = [1e-8, 1e-6]
# middle_layer_sizes = [16, 32, 64]

batch_sizes = [32]
learning_rates = [0.1]
early_stopping_thresholds = [1e-9]
middle_layer_sizes = [32]

from utils import compare_score


# for batch_size in batch_sizes:
#     for learning_rate in learning_rates:
#         for early_stopping_threshold in early_stopping_thresholds:
#             for middle_layer_size in middle_layer_sizes:


#                 layers = train_model(batch_size=batch_size, initial_learning_rate=learning_rate, early_stopping_threshold=early_stopping_threshold, middle_layer_size=middle_layer_size)


#                 with torch.no_grad():
                    
#                     x = X_train_tensor

#                     for layer in layers:
#                         x = layer(x)
#                     predictions_train = x.squeeze()
                    
                    
#                     x = X_test_tensor

#                     for layer in layers:
#                         x = layer(x)
#                     predictions_test = x.squeeze()
#                     predictions_rsquared = 1 - F.mse_loss(predictions_test, y_test_tensor) / torch.var(y_test_tensor)
                    
                    
#                     compare_score(predictions_test, y_two_year_recid_test_tensor)                    
                    
                    
                    
#                     print(f'''Batch size: {batch_size}, Learning rate: {learning_rate}, Early stopping threshold: {early_stopping_threshold}, Middle layer size: {middle_layer_size}, 
#                           train loss: {F.mse_loss(predictions_train, y_train_tensor)}, test loss: {F.mse_loss(predictions_test, y_test_tensor)}, R^2: {predictions_rsquared}''')



 
 # now we prredict the decile score using KNN using train test and validation sets above
 
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score

# k = 10 # Choose the number of neighbors
# knn = KNeighborsRegressor(n_neighbors=k)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# y_pred_train = knn.predict(X_train)
# y_pred_val = knn.predict(X_val)
# print(f'MSE on train set: {mean_squared_error(y_train, y_pred_train)}')
# print(f'MSE on validation set: {mean_squared_error(y_val, y_pred_val)}')
# print(f'MSE on test set: {mean_squared_error(y_test, y_pred)}')
# print(f'R^2 on test set: {r2_score(y_test, y_pred)}')
# print(f'R^2 on train set: {r2_score(y_train, y_pred_train)}')
# print(f'R^2 on validation set: {r2_score(y_val, y_pred_val)}')

# # we now compare_score for KNN

# compare_score(y_pred, y_two_year_recid_test_tensor)
# compare_score(y_pred_train, y_two_year_recid_train_tensor)
# compare_score(y_pred_val, y_two_year_recid_val_tensor)
 
 
# # we now prredict the decile score using Random Forest using train test and validation sets above

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score

# rf = RandomForestRegressor(n_estimators=200, random_state=42)
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# y_pred_train = rf.predict(X_train)
# y_pred_val = rf.predict(X_val)
# print(f'MSE on train set: {mean_squared_error(y_train, y_pred_train)}')
# print(f'MSE on validation set: {mean_squared_error(y_val, y_pred_val)}')
# print(f'MSE on test set: {mean_squared_error(y_test, y_pred)}')
# print(f'R^2 on test set: {r2_score(y_test, y_pred)}')
# print(f'R^2 on train set: {r2_score(y_train, y_pred_train)}')
# print(f'R^2 on validation set: {r2_score(y_val, y_pred_val)}')

# # we now compare_score for Random Forest

# compare_score(y_pred, y_two_year_recid_test_tensor)
# compare_score(y_pred_train, y_two_year_recid_train_tensor)
# compare_score(y_pred_val, y_two_year_recid_val_tensor)


# # we now prredict the decile score using Gradient Boosting using train test and validation sets above

# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score

# gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
# gb.fit(X_train, y_train)
# y_pred = gb.predict(X_test)
# y_pred_train = gb.predict(X_train)
# y_pred_val = gb.predict(X_val)
# print(f'MSE on train set: {mean_squared_error(y_train, y_pred_train)}')
# print(f'MSE on validation set: {mean_squared_error(y_val, y_pred_val)}')
# print(f'MSE on test set: {mean_squared_error(y_test, y_pred)}')
# print(f'R^2 on test set: {r2_score(y_test, y_pred)}')
# print(f'R^2 on train set: {r2_score(y_train, y_pred_train)}')
# print(f'R^2 on validation set: {r2_score(y_val, y_pred_val)}')

# # we now compare_score for Gradient Boosting

# compare_score(y_pred, y_two_year_recid_test_tensor)
# compare_score(y_pred_train, y_two_year_recid_train_tensor)
# compare_score(y_pred_val, y_two_year_recid_val_tensor)


# # we now prredict the decile score using sVM using train test and validation sets above

# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score

# svm = SVR(kernel='rbf', C=1, gamma='scale')
# svm.fit(X_train, y_train)
# y_pred = svm.predict(X_test)
# y_pred_train = svm.predict(X_train)
# y_pred_val = svm.predict(X_val)
# print(f'MSE on train set: {mean_squared_error(y_train, y_pred_train)}')
# print(f'MSE on validation set: {mean_squared_error(y_val, y_pred_val)}')
# print(f'MSE on test set: {mean_squared_error(y_test, y_pred)}')
# print(f'R^2 on test set: {r2_score(y_test, y_pred)}')
# print(f'R^2 on train set: {r2_score(y_train, y_pred_train)}')
# print(f'R^2 on validation set: {r2_score(y_val, y_pred_val)}')

# # we now compare_score for sVM

# compare_score(y_pred, y_two_year_recid_test_tensor)
# compare_score(y_pred_train, y_two_year_recid_train_tensor)
# compare_score(y_pred_val, y_two_year_recid_val_tensor)


# create a fuzzy system to predict the decile score using 'sex',  'c_charge_degree', 'age', 'priors_count' as inputs and the decile score as output with the train test and validation sets above

import numpy as np
from skfuzzy import control as ctrl
import skfuzzy as fuzz

# Create the problem variables
age = ctrl.Antecedent(np.arange(18, 70, 1), 'age')
priors_count = ctrl.Antecedent(np.arange(0, 10, 1), 'priors_count')
decile_score = ctrl.Consequent(np.arange(0, 10, 1), 'decile_score')

# Create the membership functions

age['young'] = fuzz.trimf(age.universe, [18, 18, 25])
age['middle'] = fuzz.trimf(age.universe, [20, 35, 50])
age['old'] = fuzz.trimf(age.universe, [45, 60, 60])

priors_count['low'] = fuzz.trimf(priors_count.universe, [0, 0, 2])
priors_count['medium'] = fuzz.trimf(priors_count.universe, [1, 4, 6])
priors_count['high'] = fuzz.trimf(priors_count.universe, [5, 10, 10])

# age.automf(3)
# priors_count.automf(3)
decile_score['1'] = fuzz.trimf(decile_score.universe, [0, 0, 2])
decile_score['2'] = fuzz.trimf(decile_score.universe, [1, 2, 3])
decile_score['3'] = fuzz.trimf(decile_score.universe, [2, 3, 4])
decile_score['4'] = fuzz.trimf(decile_score.universe, [3, 4, 5])
decile_score['5'] = fuzz.trimf(decile_score.universe, [4, 5, 6])
decile_score['6'] = fuzz.trimf(decile_score.universe, [5, 6, 7])
decile_score['7'] = fuzz.trimf(decile_score.universe, [6, 7, 8])
decile_score['8'] = fuzz.trimf(decile_score.universe, [7, 8, 9])
decile_score['9'] = fuzz.trimf(decile_score.universe, [8, 9, 10])
decile_score['10'] = fuzz.trimf(decile_score.universe, [9, 10, 10])

# Create the rules
rule1 = ctrl.Rule(age['young'] | priors_count['low'], decile_score['5'])
rule2 = ctrl.Rule(age['young'] | priors_count['medium'], decile_score['9'])
rule3 = ctrl.Rule(age['young'] | priors_count['high'], decile_score['10'])

rule4 = ctrl.Rule(age['middle'] | priors_count['low'], decile_score['2'])
rule5 = ctrl.Rule(age['middle'] | priors_count['medium'], decile_score['5'])
rule6 = ctrl.Rule(age['middle'] | priors_count['high'], decile_score['8'])

rule7 = ctrl.Rule(age['old'] | priors_count['low'], decile_score['2'])
rule8 = ctrl.Rule(age['old'] | priors_count['medium'], decile_score['2'])
rule9 = ctrl.Rule(age['old'] | priors_count['high'], decile_score['1'])
 
# Create the control system
decile_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# Create the simulation
decile_sim = ctrl.ControlSystemSimulation(decile_ctrl)

# Set the inputs
decile_sim.input['age'] = 25
decile_sim.input['priors_count'] = 5

# Compute the outputs
decile_sim.compute()

# Get the outputs
decile_score_value = decile_sim.output['decile_score']

print(decile_score_value)


# go through all test data and calculate MSE

y_pred = []
for i in range(len(X_test)):
    decile_sim.input['age'] = X_test[i][4]
    decile_sim.input['priors_count'] = X_test[i][0]
    decile_sim.compute()
    y_pred.append(decile_sim.output['decile_score'])

y_pred = np.array(y_pred)
y_pred = y_pred/10

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f'MSE on test set: {mse}')

print(pd.DataFrame(y_pred).describe())
print(y_test)

#stretch values of y pred to 0.1-1 from its min and max values
y_pred = (y_pred - y_pred.min())/(y_pred.max() - y_pred.min())
print(pd.DataFrame(y_pred).describe())



# use com,pare_score to compare the results

compare_score(y_pred, y_two_year_recid_test_tensor)

print(y_pred)


