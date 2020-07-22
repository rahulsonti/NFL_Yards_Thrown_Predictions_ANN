# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:40:58 2020

@author: vkaus
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
#####################################################################################################################################
# Merging datasets to train and test csv file
df_2010 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2010.csv")
df_2011 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2011.csv")
df_2012 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2012.csv")
df_2013 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2013.csv")
df_2014 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2014.csv")
df_2015 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2015.csv")
df_2016 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2016.csv")
df_2017 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2017.csv")
df_2018 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2018.csv")
df_2019 = pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\NFL-Project2019.csv")
#########################################################################################################################################
# Train data modification
df = pd.concat([pd.concat([df_2010,df_2011,df_2012,df_2013,df_2014,df_2015,df_2016,df_2017,df_2018,df_2019], axis=0)]).to_csv('ALL_Years.csv')
df =pd.read_csv(r"E:\Desktop\Masters\Semester-3\OR-603\Project-NFL\ALL_Years.csv")
list(df.columns) #Displaying list of columns
df.drop(["Unnamed: 0"],axis=1,inplace=True) #Droping unnecessary column

# Creating dummy values for specific columns
df['name'] = "dummy"
df['pass_complete'] = "dummy"
df["direction"] = "dummy"
df["other_player"] = "dummy"
df["yards_thrown"] = 0

df.dropna(subset=["Detail"], inplace=True) #Removing nas from column Detail

def test1(row):
    pattern1 = re.compile("(\w+\s*\w*) (pass complete) (\w+\s*\w*) to (\w+\s*\w*) for (\d+) .*")
    pattern2 = re.compile("(\w+\s*\w*) (pass incomplete) (\w+\s*\w*) intended for (\w+\s*\w*) .*")
    results1 = pattern1.match(row[5])
    results2 = pattern2.match(row[5])
    if results1 != None:
        results1 = results1.groups()
        row[9] = results1[0]
        row[10] = "yes"
        row[11] = results1[2]
        row[12] = results1[3]
        row[13] = results1[4]
    if results2 != None:    
        results2 = results2.groups()
        row[9] = results2[0]
        row[10] = "NO"
        row[11] = results2[2]
        row[12] = results2[3]
        row[13] = 0
    return row
   
df = df.apply(test1, 1) #Making changes to all rows in specified columns using above user defined function
df.drop(["Time","Location"],axis=1,inplace=True) #Droping 2 columns 
df.dropna(subset=["Detail","Down","ToGo"], inplace=True)# Droping nas in specific columns
df=df[~df.name.str.contains("dummy")] #Removing dummy text in name column
df = df.to_csv("NFL-DATA.csv")# Converting dataframe to csv


#################################################################################################################################################

#Building a ANN model

df = pd.read_csv('NFL-DATA.csv')
df = df.dropna()
df=df[(df.direction != 'left') & (df.direction != 'right') & (df.direction != 'middle')]
list(df.columns.values)
df = df.drop(['Unnamed: 0','Detail','name','pass_complete','other_player'],axis=1)
X = df[df["year"] <= 2018]
X.drop(["year"],axis=1,inplace=True)
X = X.values
Y = df[df["year"] == 2019]
Y.drop(["year"],axis=1,inplace=True)
Y = Y.values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X=pd.DataFrame(X)
onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()
X_train = X[:, 1:11].astype(int)
Y_train = X[:, 11].astype(int)
labelencoder_Y = LabelEncoder()
Y[:,5] = labelencoder_Y.fit_transform(Y[:,5])
Y = pd.DataFrame(Y)
onehotencoder = OneHotEncoder(categorical_features = [5])
Y = onehotencoder.fit_transform(Y).toarray()
X_test = Y[:, 1:11].astype(int)
Y_test = Y[:, 11].astype(int)




Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
   classifier = Sequential()
   classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
   classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
   classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
   classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
   return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [20,30,40],
             'epochs': [100, 200,300,400,500],
             'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#########################################################################################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=6)
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap' :[True],
    'criterion' :['mse'],
    'max_depth': [4,5,6,7,8],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [3,4,5],
    'min_samples_split': [8,10,12],
    'n_estimators': [100, 200]
}
# Create a based model

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rfr, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train, Y_train)
grid_search.best_params_


















