# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:35:07 2016

@author: ye zip
"""

import urllib.request
import os.path
from datetime import datetime as dt

# Import the Pandas library
import pandas as pd

#Import the Numpy library
import numpy as np

#Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

'''
step 1: read in data to variables
'''
print('==> Read in data to variables.', dt.now())

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train_file = 'train.csv'

if(not os.path.isfile(train_file)) :
    urllib.request.urlretrieve(train_url, train_file)
train = pd.read_csv(train_file)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test_file = 'test.csv'

if(not os.path.isfile(test_file)) :
    urllib.request.urlretrieve(test_url, test_file)

test = pd.read_csv(test_file)


'''
step 2: data cleaning and formatting
'''
print('==> Data cleaning and formatting.', dt.now())

#data cleaning anf formatting
#Convert the male and female groups to integer form
train.loc[train['Sex'] == 'male', 'Sex'] = 0
test.loc[test['Sex'] == 'male', 'Sex'] = 0
train.loc[train['Sex'] == 'female', 'Sex'] = 1
test.loc[test['Sex'] == 'female', 'Sex'] = 1

#Impute the Embarked variable
train.loc[:, 'Embarked'] = train['Embarked'].fillna('S')
test.loc[:, 'Embarked'] = test['Embarked'].fillna('S')
train.loc[:, 'Age'] = train['Age'].fillna(train['Age'].median())
test.loc[:, 'Age'] = test['Age'].fillna(test['Age'].median())

# Impute the missing value with the median
test.loc[:, 'Fare'] = test.Fare.fillna(test.Fare.median())

#Convert the Embarked classes to integer form
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
test.loc[test['Embarked'] == 'C', 'Embarked'] = 1
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 2


'''
step 3: create targets and features (feature engineering)
'''
print('==> Creating targets and features.', dt.now())

#Create the target and features numpy arrays: target, features_one
target = train['Survived'].values

#We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
feature_indexes = ['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked']
features_forest = train[feature_indexes].values


'''
step 4: control overfitting and then training
'''
print('==> Training.', dt.now())

#Building the Forest: my_forest
n_estimators = 1000
forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = n_estimators, random_state = 1)
my_forest = forest.fit(features_forest, target)

#Print the score of the random forest
print('Score of Random Forest: ', my_forest.score(features_forest, target))


'''
step 5: prediction
'''
print('==> Predicting.', dt.now())

#Compute predictions and print the length of the prediction vector:test_features, pred_forest
test_features = test[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked']].values

#Request and print the `.feature_importances_` attribute
print('Features: ', feature_indexes)
print('Feature Importance: ', my_forest.feature_importances_)


'''
step 6: save results
'''
print('==> Saving results.', dt.now())

# Make your prediction using the test set
my_prediction = my_forest.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test['PassengerId']).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ['Survived'])
#print(my_solution)

# Check that your data frame has 418 entries
print('Size of Solution: ', my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv('my_solution_one.csv', index_label = ['PassengerId'])


