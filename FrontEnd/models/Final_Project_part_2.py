import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split,StratifiedShuffleSplit,StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as imbpipeline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pathlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import scipy.sparse
from sklearn.metrics import accuracy_score,confusion_matrix


###############################################################################
#                        DATA Pre-Processing and Modeling
###############################################################################

class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self 

class nanReplacer():
    def __init__(self,columns,value):
        self.columns=columns
        self.value=value

    def transform(self,X,y=None):
        X[self.columns]=X[self.columns].replace(np.nan,self.value)
        return X

    def fit(self, X, y=None):
        return self 

data=pd.read_csv("KSI.csv") #data imported in dataframe

data = data[data['ACCLASS'].notna()]        #dropped rows where target is Null
data["ACCLASS"][data["ACCLASS"]=='Property Damage Only']='Non-Fatal Injury'     #Replaced property damage to Non Fatal in target column
data["ACCLASS"][data["ACCLASS"]=='Non-Fatal Injury']='Non-Fatal'        #Replaced property Non Fatal Injury to Non Fatal in target column

#list of relevent columns
relevent_columns=['ACCLASS','AG_DRIV', 'ALCOHOL', 'AUTOMOBILE', 'CYCLIST', 'DISABILITY', 'DRIVACT', 'DRIVCOND', 'EMERG_VEH', 'IMPACTYPE', 'INJURY', 'INVAGE', 'LIGHT', 'MOTORCYCLE', 'PASSENGER', 'PEDESTRIAN', 'RDSFCOND', 'REDLIGHT', 'ROAD_CLASS', 'SPEEDING', 'TRAFFCTL', 'TRSN_CITY_VEH', 'TRUCK', 'VISIBILITY']

#list of columns which are not relevent
drop_columns_list=[x for x in data.columns.values if x not in relevent_columns]

#pipeline to drop columns
dropColumnsPipeline = Pipeline([
                                ("columnDropper", columnDropperTransformer(drop_columns_list))
                                ])

#list of columns with yes and no type of answers
lis=['ACCLASS', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']

#list of columns which has "Other" in answers
lis_with_other=['ROAD_CLASS', 'VISIBILITY', 'RDSFCOND', 'IMPACTYPE', 'DRIVCOND']



# Data cleaning  pipeline
data=dropColumnsPipeline.fit_transform(data)

replacePipeline=Pipeline([
                        ("nanToNOReplace",nanReplacer(lis,'No')),   #replace null values to NO
                        ("nanToOtherReplace",nanReplacer(lis_with_other,'Other')),  #replace null values to Other
                        ("nanToNoneReplace",nanReplacer(['INJURY'],'None')),        #replace null values to 'None'
                        ("nanToNoControlReplace",nanReplacer(['TRAFFCTL','DRIVACT'],'Unknown'))   #replace null values to Unknown
                        ])

data=replacePipeline.fit_transform(data)

#exploring data after transformation

print(data.head())

print(data.columns.values)

print(data.shape)

print((data.info()))


###############################################################################
#                        Model training and testing
###############################################################################
data=data.astype("category")

target=["ACCLASS"]

features=np.setdiff1d(data.columns.values,target).tolist()

feature_data=data[features]

target_data=data[target]

transformed_feature_data,target_data=feature_data,target_data

splitter=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=12)

for train,test in splitter.split(transformed_feature_data,target_data):     #this will splits the index
    X_train = transformed_feature_data.iloc[train]
    y_train = target_data.iloc[train]
    X_test = transformed_feature_data.iloc[test]
    y_test = target_data.iloc[test]

#data transformation from catagorical to numeric and scaled
transformPipeline = Pipeline([
                                ("OneHot", OneHotEncoder()),
                                ("scaler", StandardScaler(with_mean=False))
                                ])

transformPipeline.fit(X_train,y_train)
X_train_transformed=transformPipeline.transform(X_train)
X_test_transformed=transformPipeline.transform(X_test)

pkl_file = open("LogisticRegressionModel.pkl","rb")

classifier = joblib.load(pkl_file)

pkl_file.close()

y_pred = classifier.predict(X_test_transformed)


print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

#pipeline to oversamle the imbalanced calss and model
clfPipeline=imbpipeline([
                          ('smote',SMOTE(k_neighbors=3)),  
                        ('logisticRegression',LogisticRegression())
                    ])

stratified_kfold = StratifiedKFold(n_splits=3,
                                       shuffle=True,
                                       random_state=11)

#grid search for finding best parameter
param_grid = {'logisticRegression__C':[0.00001,0.0001,0.001],
              'logisticRegression__solver':['lbfgs','saga'],
              'logisticRegression__max_iter':[1000,500,750]}
grid_search = GridSearchCV(estimator=clfPipeline,
                           param_grid=param_grid,
                           cv=stratified_kfold
                           )

grid_search.fit(X_train_transformed, y_train)
cv_score = grid_search.best_score_
test_score = grid_search.score(X_test_transformed, y_test)
print(grid_search.best_estimator_)
joblib.dump(grid_search.best_estimator_, "LogisticRegressionModel.pkl")
print(f'\n\nCross-validation score: {cv_score}\nTest score: {test_score}')



clfPipeline=imbpipeline([
                          ('smote',SMOTE(k_neighbors=3)),  
                        ('DecisionTree',DecisionTreeClassifier())
                    ])

stratified_kfold = StratifiedKFold(n_splits=3,
                                       shuffle=True,
                                       random_state=11)

#grid search for finding best parameter
param_grid = {'DecisionTree__criterion':['gini', 'entropy', 'log_loss'],
              'DecisionTree__max_depth':[10,15,20,30,40]}
grid_search = GridSearchCV(estimator=clfPipeline,
                           param_grid=param_grid,
                           cv=stratified_kfold
                           )

grid_search.fit(X_train_transformed, y_train)
cv_score = grid_search.best_score_
test_score = grid_search.score(X_test_transformed, y_test)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
joblib.dump(grid_search.best_estimator_, "DecisionTreeModel.pkl")

print(f'\n\nCross-validation score: {cv_score}\nTest score: {test_score}')


clfPipeline=imbpipeline([
                          ('smote',SMOTE(k_neighbors=3)),  
                        ('svm',SVC(max_iter=1000))
                    ])

stratified_kfold = StratifiedKFold(n_splits=3,
                                       shuffle=True,
                                       random_state=11)

#grid search for finding best parameter
param_grid = {'svm__kernel':['linear', 'poly', 'rbf', 'sigmoid'],
              'svm__C':[1,0.1,10]}
grid_search = GridSearchCV(estimator=clfPipeline,
                           param_grid=param_grid,
                           cv=stratified_kfold
                           )

grid_search.fit(X_train_transformed, y_train)
cv_score = grid_search.best_score_
test_score = grid_search.score(X_test_transformed, y_test)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
joblib.dump(grid_search.best_estimator_, "SVMModel.pkl")

print(f'\n\nCross-validation score: {cv_score}\nTest score: {test_score}')


clfPipeline=imbpipeline([
                          ('smote',SMOTE(k_neighbors=3)),  
                        ('forest',RandomForestClassifier())
                    ])

stratified_kfold = StratifiedKFold(n_splits=3,
                                       shuffle=True,
                                       random_state=11)

#grid search for finding best parameter
param_grid = {'forest__n_estimators':[10,20,30,40],
              'forest__criterion':['gini', 'entropy', 'log_loss'],
              'forest__max_depth':[20,40,10]}
grid_search = GridSearchCV(estimator=clfPipeline,
                           param_grid=param_grid,
                           cv=stratified_kfold
                           )

grid_search.fit(X_train_transformed, y_train)
cv_score = grid_search.best_score_
test_score = grid_search.score(X_test_transformed, y_test)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
joblib.dump(grid_search.best_estimator_, "RandomForestModel.pkl")

print(f'\n\nCross-validation score: {cv_score}\nTest score: {test_score}')


neural_network_model = Sequential()
neural_network_model.add(Dense(64, activation='relu', input_dim=X_train_transformed.shape[1]))  # Adjust input_dim to match your feature dimensions
neural_network_model.add(Dense(32, activation='relu'))
neural_network_model.add(Dense(32, activation='relu'))
neural_network_model.add(Dense(1, activation='sigmoid'))  # For binary classification, adjust the output layer neurons accordingly
neural_network_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

X_train_transformed=pd.DataFrame.sparse.from_spmatrix(X_train_transformed)
X_test_transformed=pd.DataFrame.sparse.from_spmatrix(X_test_transformed)
y_train=y_train.replace('Fatal',1)
y_train=y_train.replace('Non-Fatal',0)
y_test=y_test.replace('Fatal',1)
y_test=y_test.replace('Non-Fatal',0)
neural_network_model.fit(X_train_transformed, y_train, epochs=50)


predict=neural_network_model.predict(X_test_transformed).round()

print(accuracy_score(y_test,predict))
print(confusion_matrix(y_test,predict))

joblib.dump(neural_network_model, "NeuralNetworkModel.pkl")

neural_network_model.save('my_model.keras')

new_model = tf.keras.models.load_model('my_model.keras')

# Show the model architecture
new_model.summary()

predict=new_model.predict(X_test_transformed).round()

print(accuracy_score(y_test,predict))
print(confusion_matrix(y_test,predict))


