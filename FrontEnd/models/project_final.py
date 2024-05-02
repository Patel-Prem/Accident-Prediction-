
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:36:49 2023

@author: Group 7
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel, chi2
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

# Load the data
data = pd.read_csv('D:/pravalika/Centennial college/semester 2/comp 247 Supervised learning/project/KSI.csv')
pd.set_option('display.max_columns', None)
data.info()
data.describe()
data.isna().sum()
data.corr()
path = r'D:\pravalika\Centennial college\semester 2\comp 247 Supervised learning\project\project_files'

print("Summary Statistics:")
print()
# Initialize the table headers
headers = ["Column Name", "Column Type", "Range", "Example Values"]
rows = []
column_types = data.dtypes

for column in data.columns:
    column_name = column
    column_type = str(column_types[column])
    # Extract range and example values based on column type
    if column_type == 'int64' or column_type == 'float64':
        column_range = (data[column].min(), data[column].max())
        example_values = ', '.join(map(str, data[column].sample(5).values))
    else:
        column_range = "NA"
        example_values = ', '.join(map(str, data[column].sample(5).values))
        
    rows.append([column_name, column_type, column_range, example_values])


table = tabulate(rows, headers, tablefmt="pipe")
print(table)

#Statistical assessments including means, averages, correlations
mean = data.mean()
median = data.median()
correlation_matrix = data.corr()

print("Column Means:")
print(mean)
print("\nColumn Medians:")
print(median)
print("\nCorrelation Matrix:")
print(correlation_matrix)

#Missing data evaluations – use pandas, numpy and any other python packages
missing_values = data.isnull().sum()
print(missing_values)

#Graphs and visualizations – use pandas, matplotlib, seaborn, numpy and any other python packages, you also can use power BI desktop.
import matplotlib.pyplot as plt
import seaborn as sns


#Graph-1 
data.hist(bins=50, figsize=(20,15))
plt.show()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


#Graph-2 : Traffic Collisions per Year
total_2006, total_2007, total_2008, total_2009, total_2010, total_2011 = 0, 0, 0, 0, 0, 0
total_2012, total_2013, total_2014, total_2015, total_2016, total_2017, total_2018 = 0, 0, 0, 0, 0, 0, 0
total_2019, total_2020, total_2021, total_2022, total_2023 = 0, 0, 0, 0, 0

for value in data['YEAR']:
    if value == 2006:
        total_2006 += 1
    elif value == 2007:
        total_2007 += 1
    elif value == 2008:
        total_2008 += 1
    elif value == 2009:
        total_2009 +=1
    elif value == 2010:
        total_2010 +=1
    elif value == 2011:
        total_2011 +=1
    elif value == 2012:
        total_2012 +=1
    elif value == 2013:
        total_2013 +=1
    elif value == 2014:
        total_2014 +=1
    elif value == 2015:
        total_2015 +=1
    elif value == 2016:
        total_2016 +=1
    elif value == 2017:
        total_2017 +=1
    elif value == 2018:
        total_2018 +=1
    elif value == 2019:
        total_2019 +=1
    elif value == 2020:
        total_2020 +=1
    elif value == 2021:
        total_2021 +=1
    elif value == 2022:
        total_2022 +=1
    elif value == 2023:
        total_2023 +=1

x = [total_2006, total_2007, total_2008,total_2009,total_2010,total_2011,total_2012,total_2013,total_2014,total_2015,total_2016,total_2017,total_2018,total_2019,total_2020,total_2021,total_2022,total_2023]
years = range(2006, 2024)

# Create the bar graph
plt.bar(years, x)

# Set the x-axis tick labels
plt.xticks(years,rotation=45)

# Add labels and title
plt.title('Traffic Collisions per Year')
plt.xlabel('Years')
plt.ylabel('Traffic Collisions')

# Show the plot
plt.show()


# Graph-3: Collision Location vs INJURY
data = data.dropna(subset=['INJURY'])
data['INJURY']
missing_values = data['INJURY'].isnull().sum()

print(missing_values)

print(data['INJURY'].head(20))

data['INJURY'] = data['INJURY'].map({'None': 0, 'Non-Fatal Injury': 0.25, 'Minor': 0.50, 'Major': 0.75, 'Fatal': 1})
data['INJURY']

#Since there is geographical information (latitude and longitude), it is a good idea to create a scatterplot of all districts to visualize the data
scatter_plot_INJURY=data.plot(kind="scatter", x="LONGITUDE", y="LATITUDE",  
          c=data["INJURY"], colorbar=True,
          label="Collision Location vs Injury", 
          cmap=plt.get_cmap("jet"),
          figsize=(10,7))

# Add a label to the color bar
color_bar = scatter_plot_INJURY.collections[0].colorbar

tick_locations = np.linspace(0, 1, num=5)
tick_labels = ['None', 'Non-Fatal Injury', 'Minor', 'Major', 'Fatal']
color_bar.set_ticks(tick_locations)
color_bar.set_ticklabels(tick_labels)

#Graph-4 Toronto Areas vs  Number od accidents.
area = pd.DataFrame()
area['Etobicoke'] = data['YEAR'][data['DISTRICT']=='Etobicoke York'].value_counts()
area['NorthYork'] = data['YEAR'][data['DISTRICT']=='North York'].value_counts()
area['Scarborough'] = data['YEAR'][data['DISTRICT']=='Scarborough'].value_counts()
area['Toronto&EastYork'] = data['YEAR'][data['DISTRICT']=='Toronto and East York'].value_counts()
area.loc['Total']= area.sum()
ig, ax = plt.subplots(1, 1, figsize=(10, 8))
area.iloc[-1].plot(kind='bar', ax=ax)
ax.set_ylabel('Number of Accidents', fontsize=14)
ax.set_xlabel('Toronto Areas', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.title('Toronto Areas vs Number of accidents', fontsize=16)
plt.show()


#Null percentage for column PEDTYPE - 81.54%, PEDACT - 81.41%, PEDCOND - 81.41%, CYCLISTYPE - 95.33%, CYCACT - 95.37%, CYCCOND - 95.38%, OFFSET - 79.60%, FATAL_NO - 95.01%
columns_to_drop = ['FATAL_NO', 'INJURY','INITDIR','VEHTYPE','MANOEUVER','PEDTYPE','PEDACT', 
                   'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND', 'X', 'Y', 'INDEX_', 'ACCNUM', 
                   'WARDNUM', 'DIVISION', 'ObjectId', 'OFFSET', 'INVAGE']

data.drop(columns_to_drop, axis=1, inplace=True)
replace_columns = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH',
                      'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']
data[replace_columns] = data[replace_columns].replace('Yes', 1).fillna(0)
data['ACCLOC'].unique()
data['DRIVCOND'].unique()
data['DRIVACT'].unique()
data['IMPACTYPE'].unique()
data['ACCLASS'].unique()
data['TRAFFCTL'].unique()
data['LOCCOORD'].unique()

def DRIVCOND_REPLACE(data):
    replace_drivcond = {
        'Normal': 0,
        'Ability Impaired, Alcohol Over .08': 1,
        'Inattentive': 1,
        'Fatigue': 1,
        'Other': 1,
        'Had Been Drinking': 1,
        'Ability Impaired, Drugs': 1,
        'Medical or Physical Disability': 1,
        'Ability Impaired, Alcohol': 1,
        'Unknown': np.nan
    }
    return data.replace(replace_drivcond)

def DRIVACT_REPLACE(data):
    replace_drivact = {
        'Driving Properly': 1,
        'Exceeding Speed Limit': 0,
        'Disobeyed Traffic Control': 0,
        'Following too Close': 0,
        'Lost control': 0,
        'Failed to Yield Right of Way': 0,
        'Improper Passing': 0,
        'Improper Turn': 0,
        'Other': 0,
        'Speed too Fast For Condition': 0,
        'Improper Lane Change': 0,
        'Wrong Way on One Way Road': 0,
        'Speed too Slow': 0,
    }
    return data.replace(replace_drivact)

def ACCCLASS_REPLACE(data):
    replace_acclass = {
        'Non-Fatal Injury': 0,
        'Fatal': 1,
        'Property Damage Only': 0
    }
    return data.replace(replace_acclass)

def invtype_replace(data):
    driver_types = ['Driver', 'Truck Driver', 'Motorcycle Driver', 'Moped Driver']
    pedestrian_types = ['Pedestrian', 'Wheelchair', 'Pedestrian - Not Hit']
    cyclist_types = ['Cyclist', 'In-Line Skater', 'Cyclist Passenger']
    
    if data in driver_types:
        return 'Driver'
    elif data in pedestrian_types:
        return 'Pedestrian'
    elif data in cyclist_types:
        return 'Cyclist'
    else:
        return 'Other'
  
def ROADCLASSS_REPLACE(data):
    replace_roadclass = {
        'Minor Arterial': 'Arterial',
        'Collector': 'Collector',
        'Major Arterial': 'Arterial',
        'Local': 'Local',
        'Expressway': 'Expressway',
        'Expressway Ramp': 'Expressway',
        'Major Arterial Ramp': 'Arterial',
        'Other': 'Other',
        'Pending': 'Other',
        'Laneway': 'Other',
        'unknown': 'Other'
    }
    return data.replace(replace_roadclass)

def traffctl_replace(data):
    replace_traffctl = {
        'No Control': 0,
        'Stop Sign': 1,
        'Yield Sign': 1,
        'School Guard': 1,
        'Traffic Gate': 1,
        'Police Control': 1,
        'Streetcar (Stop for)': 1,
        'Traffic Signal': 1,
        'Pedestrian Crossover': 1,
        'Traffic Controller': 1
    }
    return data.replace(replace_traffctl)

def replace_loccoord(data):
    replace_loccoord = {
        'Exit Ramp Southbound': 0,
        'Mid-Block (Abnormal)': 0,
        'Intersection': 1,
        'Mid-Block': 0,
        'Park, Private Property, Public Lane': 0,
        'Exit Ramp Westbound': 1,
        'Entrance Ramp Westbound': 0
    }
    return data.replace(replace_loccoord)

def dayschedule(value):
    value = int(value)
    if value >= 0 and value < 1200:
        return 'Morning'
    elif value >= 1200 and value <= 1700:
        return 'Afternoon'
    else:
        return 'Night'

data['DRIVING_CONDITION'] = DRIVCOND_REPLACE(data['DRIVCOND'])
data.drop(['DRIVCOND'], axis=1, inplace=True)

data['DRIVE_ACTION'] = DRIVACT_REPLACE(data['DRIVACT'])
data.drop(['DRIVACT'], axis=1, inplace=True)

data.dropna(subset=['ACCLASS'], inplace=True)
data['ACCLASS'] = ACCCLASS_REPLACE(data['ACCLASS'])

# Transform invtype
data['INVTYPE'] = data['INVTYPE'].apply(invtype_replace)

# Transform road class
data['ROAD_CLASS'] = ROADCLASSS_REPLACE(data['ROAD_CLASS'])

# Transform tarffctl
data['TRAFFCTL'] = traffctl_replace(data['TRAFFCTL'])

# Transform loccoord
data['INTERSECTION'] = replace_loccoord(data['LOCCOORD'])
data.drop(['LOCCOORD'], axis=1, inplace=True)

# Split date into month and day columns
data['DATE'] = pd.to_datetime(data['DATE'], format='%Y/%m/%d %H:%M:%S')
data['MONTH'] = data['DATE'].dt.month
data['DAY'] = data['DATE'].dt.day
data.drop('DATE', axis=1, inplace=True)

data['TIME'] = data['TIME'].apply(dayschedule)

print(data['HOOD_140'].value_counts(dropna=False))
print(data['HOOD_158'].value_counts(dropna=False))
#NEIGHBOURHOOD_158,NEIGHBOURHOOD_140 is same as HOOD_140,HOOD_158
data = data.drop(columns=['NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140', 'STREET1', 'STREET2'])
# Reset index
data.reset_index(drop=True, inplace=True)

'''
Train-Test Split
'''
X = data.drop(['ACCLASS'], axis=1)
y = data['ACCLASS']

StratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=76) #splt the data 20% test

for train_index, test_index in StratifiedShuffleSplit.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

def preprocessing_pipeline(data):
    numerical_columns = list(data.select_dtypes(include=np.number).columns)
    category_columns = list(data.select_dtypes(include=object).columns)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    final_pipeline = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, category_columns)
        ], verbose_feature_names_out=False)
    
    return final_pipeline

'''
Feaature Importance & Model Training
'''
clf = ImbPipeline(steps=
                    [('full_pipeline', preprocessing_pipeline(X_train)),
                        ('smote', BorderlineSMOTE(random_state=76)),
                        ('variance', VarianceThreshold(0.1)),
                        ('feature_selection', SelectKBest(chi2, k=10)),
                        ('classifier', RandomForestClassifier(random_state=76))
                    ])

clf.fit_resample(X_train, y_train)
clf.fit(X_train, y_train)

importances = clf.named_steps['classifier'].feature_importances_
feature_names = clf.named_steps['full_pipeline'].transformers_[0][2] + \
                list(clf.named_steps['full_pipeline'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())
feature_names = feature_names[:len(importances)]
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)

feature_importances.sort_values(by='importance', ascending=False, inplace=True)
# Calculate total importance for normalization
total_importance = feature_importances['importance'].sum()
feature_importances['normalized_importance'] = feature_importances['importance'] / total_importance

plt.figure(figsize=(8, 8))
plt.pie(
    feature_importances['normalized_importance'],
    labels=feature_importances['feature'],
    autopct=lambda p: '{:.1f}%'.format(p) if p > 0.5 else '',
    startangle=140
)
plt.title('Feature Importance (Pie Chart)')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()


print("Train Accuracy:", accuracy_score(y_train, clf.predict(X_train))) #training accuracy
print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test))) #testing accuracy

X_train = X_train[feature_importances['feature']]
X_test = X_test[feature_importances['feature']]

data.columns

'''
PART B
'''
classifiers = {
    'XGBoost': XGBClassifier(random_state=76),
    'SVM': SVC(random_state=76),
    'Decision Tree': DecisionTreeClassifier(random_state=76),
    'Random Forest': RandomForestClassifier(random_state=76),
    'Logistic Regression': LogisticRegression(random_state=76)
}

param_grids = {
    'XGBoost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.1, 0.01, 0.05]
    },
    'SVM': {
        'classifier__kernel': ['linear', 'rbf', 'poly'],
        'classifier__C': [0.01, 0.1, 0.5, 1],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__degree': [2, 3]
    },
    'Decision Tree': {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [5, 10, 15, 20, 25, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'Logistic Regression': {
        'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'classifier__C': [0.01, 0.1, 0.5, 1],
        'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'classifier__max_iter': [100, 200, 300, 400, 500]
    }
}

new_pre_processing_pipeline = preprocessing_pipeline(X_train)

for classifier_name, classifier in classifiers.items():
    print(f"Training {classifier_name}...")
    

    param_grid = param_grids[classifier_name]
 
    pipeline = ImbPipeline(steps=
                        [('new_pipeline', new_pre_processing_pipeline),
                            ('classifier', classifier)
                        ])
 
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=10,
                               cv=5)
    
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    print(f"Best parameters {best_params}")
    print(f"Best {classifier_name} Train Accuracy:", accuracy_score(y_train, best_estimator.predict(X_train)))
    print(f"Best {classifier_name} Test Accuracy:", accuracy_score(y_test, best_estimator.predict(X_test)))
    print("Confusion matrix \n", confusion_matrix(y_test, best_estimator.predict(X_test)))
    print("Classification Report \n", classification_report(y_test, best_estimator.predict(X_test)))
    
    joblib.dump(best_estimator, path+f'\{classifier_name.lower().replace(" ", "_")}_best_estimator.pkl')
