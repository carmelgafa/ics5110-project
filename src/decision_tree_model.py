import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
label_encoder = LabelEncoder()
scaler = StandardScaler()
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings(action='ignore')


df_compas_path = os.path.join('data', 'compas-scores-two-years.csv')
df_compas = pd.read_csv(df_compas_path)





df_compas = df_compas.drop(['last','first','out_custody','in_custody','c_offense_date','decile_score.1',
                      'priors_count.1','c_case_number','days_b_screening_arrest','start','end',
                      'event','screening_date','c_case_number','r_days_from_arrest','id',
                    #   'r_charge_degree',
                      'r_offense_date',
                      'vr_case_number','r_case_number','r_jail_out','c_arrest_date',
                    #   'r_charge_desc',
                      'r_jail_in', 'violent_recid','vr_charge_degree','vr_offense_date','vr_charge_desc'], axis=1)




# Define the refined offence mapping
offence_mapping = {
    # Violence-related offences
    'Battery': 'Violence',
    'Felony Battery': 'Violence',
    'Batt': 'Violence',
    'Assault': 'Violence',
    'Stalking': 'Violence',
    'Abuse': 'Violence',
    'Aggress': 'Violence',
    'Threat': 'Violence',
    'Fight': 'Violence',
    'Resist': 'Violence',
    'Obstruction': 'Violence',
    'Harassment': 'Violence',
    'Intimidation': 'Violence',
    'Murder': 'Violence',
    'Homicide': 'Violence',

    # Disorder and Minor Crimes
    'Disorderly': 'Disorder',
    'Escape': 'Disorder',
    'Contempt': 'Disorder',
    'Violation': 'Disorder',
    'Arrest': 'Disorder',
    'Loiter': 'Disorder',
    'Curfew': 'Disorder',
    'Minor': 'Disorder',
    'Breach': 'Disorder',
    'Mischief': 'Disorder',

    # Theft-related offences
    'Theft': 'Theft',
    'Larceny': 'Theft',
    'Robbery': 'Theft',
    'Burglary': 'Theft',
    'Shoplifting': 'Theft',
    'Auto': 'Theft',
    'Stolen': 'Theft',
    'Grand Theft': 'Theft',
    'Petit Theft': 'Theft',

    # Fraud-related offences
    'Fraud': 'Fraud',
    'Forge': 'Fraud',
    'Forgery': 'Fraud',
    'Embezzle': 'Fraud',
    'Tamper': 'Fraud',
    'Credit': 'Fraud',
    'Identity': 'Fraud',
    'Check': 'Fraud',
    'False': 'Fraud',
    'Uttering': 'Fraud',
    'Forged': 'Fraud',
    'Misrepresent': 'Fraud',

    # Drug-related offences
    'Possession': 'Intoxication',
    'Possess': 'Intoxication',
    'Cannabis': 'Intoxication',
    'Marijuana': 'Intoxication',
    'Cocaine': 'Intoxication',
    'Heroin': 'Intoxication',
    'Meth': 'Intoxication',
    'Narcotics': 'Intoxication',
    'Drug': 'Intoxication',
    'Controlled Substance': 'Intoxication',
    'Paraphernalia': 'Intoxication',
    'Subst': 'Intoxication',
    'Contr Subst': 'Intoxication',
    'Drunk': 'Intoxication',
    'Alcoholic': 'Intoxication',
    'Intoxicated': 'Intoxication',
    'Poss' : 'Intoxication',
    'Alcohol': 'Intoxication',
    'Drinking': 'Intoxication',
    'Traffick': 'Intoxication',

    # Traffic-related offences
    'Driving': 'Traffic',
    'Drvng': 'Traffic',
    'Drivng': 'Traffic',
    'Drv': 'Traffic',
    'License': 'Traffic',
    'Lic': 'Traffic',
    'DUI': 'Traffic',
    'Susp': 'Traffic',
    'Suspended': 'Traffic',
    'Reckless': 'Traffic',
    'Speeding': 'Traffic',
    'Revoked': 'Traffic',
    'Operating': 'Traffic',
    'Vehicle': 'Traffic',
    'Veh': 'Traffic',
    'Registration': 'Traffic',
    'Expired DL': 'Traffic',

    # Morality-related offences
    'Indecent': 'Morality',
    'Loitering': 'Morality',
    'Prostitution': 'Morality',
    'Lewd': 'Morality',
    'Obscene': 'Morality',
    'Voyeur': 'Morality',
    'Exposure': 'Morality',

    # Arson-related offences
    'Arson': 'Arson',
    'Vandalism': 'Arson',
    'Trespass': 'Arson',
    'Damage': 'Arson',
    'Graffiti': 'Arson',

    # Weapon-related offences
    'Weapon': 'Weapons',
    'Firearm': 'Weapons',
    'Gun': 'Weapons',
    'Explosives': 'Weapons',
    'Knife': 'Weapons',
    'Ammo': 'Weapons',
    'Rifle': 'Weapons',


    # Domestic-related offences
    'Domestic': 'Domestic',
    'Spouse': 'Domestic',
    'Family': 'Domestic',
    'Child Abuse': 'Domestic',
    'Neglect': 'Domestic',

    # Relapse-related offences
    'Viol Pretrial': 'Relapse',
    'Pretrial Release': 'Relapse',
    'Release Dom': 'Relapse',
    'Injunc Repeat': 'Relapse',
    'Repeat Viol': 'Relapse',
    'Extradition/Defendants': 'Relapse',

    # Unresolved and other offences
    'Case No Charge': 'Unresolved',
    'Arrest Case': 'Unresolved'
}

# Function to assign categories
def assign_category(description):
    if pd.isna(description):
        return 'Other'
    for key, category in offence_mapping.items():
        if key.lower() in description.lower():
            return category
    
    print(description)
    return 'Other'

# Create new columns for c_charge_cat and r_charge_cat
df_compas['c_charge_cat'] = df_compas['c_charge_desc'].apply(assign_category)
df_compas['r_charge_cat'] = df_compas['r_charge_desc'].apply(assign_category)




df_compas['c_jail_in'] = pd.to_datetime(df_compas['c_jail_in'])
df_compas['c_jail_out'] = pd.to_datetime(df_compas['c_jail_out'])
df_compas['days_in_jail'] = abs((df_compas['c_jail_out'] - df_compas['c_jail_in']).dt.days)


df_compas['days_in_jail'].describe()

df_compas['compas_screening_date'] = pd.to_datetime(df_compas['compas_screening_date'])

df_compas['v_screening_date'] = pd.to_datetime(df_compas['v_screening_date'])


# Impute missing values for numerical variables
numeric_cols = ['c_days_from_compas', 'v_decile_score']
for col in numeric_cols:
    df_compas[col].fillna(df_compas[col].median(), inplace=True)

# Impute missing values for categorical variables
categorical_cols = ['c_charge_degree', 'score_text', 'v_score_text', 'c_jail_in', 'c_jail_out', 'c_charge_desc', 'days_in_jail']
for col in categorical_cols:
    df_compas[col].fillna(df_compas[col].mode()[0], inplace=True)

# Check if there are any missing values remaining
print(df_compas.isnull().sum())

df_compas["sex"].replace({'Male': 1, 'Female': 0}, inplace=True)
df_compas["c_charge_degree"].replace({'F': 1, 'M': 0}, inplace=True)
df_compas["r_charge_degree"].replace({'F': 1, 'M': 0}, inplace=True)
df_compas['c_charge_cat'] = label_encoder.fit_transform(df_compas['c_charge_cat'])
# df_compas['r_charge_cat'] = label_encoder.fit_transform(df_compas['r_charge_cat'])

recid = df_compas['is_recid']
violent_recid = df_compas['is_violent_recid']

X = df_compas[['age', 'sex', 'priors_count', 'race',  'c_charge_degree']]

print(X.head())

Y = df_compas['two_year_recid']


X_encoded = X.copy()
X_encoded['race_encoded'] = label_encoder.fit_transform(X['race'])
X_encoded.drop(['race'], axis=1, inplace=True)


X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)

model.fit(X_train_scaled, Y_train)

Y_pred_random = model.predict(X_test_scaled)

print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_random))
print("Classification Report:\n", classification_report(Y_test, Y_pred_random))


# let us try with a knn model

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train_scaled, Y_train)
Y_pred_knn = model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_knn))
print("Classification Report:\n", classification_report(Y_test, Y_pred_knn))


# let us try with logistic regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)
Y_pred_lr = model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_lr))
print("Classification Report:\n", classification_report(Y_test, Y_pred_lr))


# let us try with a neural network

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(random_state=42, hidden_layer_sizes=(100, ), max_iter=1000)
model.fit(X_train_scaled, Y_train)
Y_pred_mpl = model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_mpl))
print("Classification Report:\n", classification_report(Y_test, Y_pred_mpl))

# let us try an svm model

from sklearn.svm import SVC
model = SVC(random_state=42, C=1, kernel='rbf', gamma='scale')
model.fit(X_train_scaled, Y_train)
Y_pred_svc = model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_svc))
print("Classification Report:\n", classification_report(Y_test, Y_pred_svc))

# let us try a decision tree

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=10)
model.fit(X_train_scaled, Y_train) 
Y_pred_decision = model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_decision))
print("Classification Report:\n", classification_report(Y_test, Y_pred_decision))


# create roc auc curve for all models

from sklearn.metrics import roc_curve, auc
fpr_random, tpr_random, _ = roc_curve(Y_test, Y_pred_random)
fpr_knn, tpr_knn, _ = roc_curve(Y_test, Y_pred_knn)
fpr_lr, tpr_lr, _ = roc_curve(Y_test, Y_pred_lr)
fpr_mpl, tpr_mpl, _ = roc_curve(Y_test, Y_pred_mpl)
fpr_svc, tpr_svc, _ = roc_curve(Y_test, Y_pred_svc)
fpr_decision, tpr_decision, _ = roc_curve(Y_test, Y_pred_decision)

roc_auc_random = auc(fpr_random, tpr_random)
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_mpl = auc(fpr_mpl, tpr_mpl)
roc_auc_svc = auc(fpr_svc, tpr_svc)
roc_auc_decision = auc(fpr_decision, tpr_decision)

plt.figure()
lw = 2
plt.plot(fpr_random, tpr_random, color='darkorange',
         lw=lw, label='Random (area = %0.2f)' % roc_auc_random)
plt.plot(fpr_knn, tpr_knn, color='green',
         lw=lw, label='KNN (area = %0.2f)' % roc_auc_knn)
plt.plot(fpr_lr, tpr_lr, color='blue',
         lw=lw, label='Logistic Regression (area = %0.2f)' % roc_auc_lr)
plt.plot(fpr_mpl, tpr_mpl, color='red',
         lw=lw, label='Neural Network (area = %0.2f)' % roc_auc_mpl)
plt.plot(fpr_svc, tpr_svc, color='yellow',
         lw=lw, label='SVM (area = %0.2f)' % roc_auc_svc)
plt.plot(fpr_decision, tpr_decision, color='black',
         lw=lw, label='Decision Tree (area = %0.2f)' % roc_auc_decision)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# do a grid search for the decision tree model

from sklearn.model_selection import GridSearchCV
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_scaled, Y_train)
print(grid_search.best_params_)


# do grid search for the svm model

param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'poly'],
              'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_scaled, Y_train)
print(grid_search.best_params_)


# do grid search for the neural network model

param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100), (50, 50, 50), (100, 100, 100)],
                'max_iter': [1000, 2000, 3000]}
grid_search = GridSearchCV(MLPClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_scaled, Y_train)
print(grid_search.best_params_)
