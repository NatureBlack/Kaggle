# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:35:07 2016

@author: ye zip
"""

import urllib.request
import os.path

# Import the Pandas library
import pandas as pd

#Import the Numpy library
import numpy as np

#Import 'tree' from scikit-learn library
from sklearn import tree

'''
step 1: read in data to variables
'''

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

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())

# Normalized male survival
print(train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True))

# Normalized female survival
print(train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True))


'''
step 2: data cleaning and formatting
'''


#data cleaning anf formatting
#Convert the male and female groups to integer form
train['Sex'].loc[train['Sex'] == 'male'] = 0
train['Sex'].loc[train['Sex'] == 'female'] = 1
test['Sex'].loc[test['Sex'] == 'male'] = 0
test['Sex'].loc[test['Sex'] == 'female'] = 1

#Impute the Embarked variable
train["Embarked"] = train['Embarked'].fillna('S')
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Embarked"] = test['Embarked'].fillna('S')
test["Age"] = test["Age"].fillna(test["Age"].median())
# Impute the missing value with the median
test.Fare[152] = test['Fare'].median()

#Convert the Embarked classes to integer form
train["Embarked"].loc[train["Embarked"] == "S"] = 0
train['Embarked'].loc[train['Embarked'] == 'C'] = 1
train['Embarked'].loc[train['Embarked'] == 'Q'] = 2
test["Embarked"].loc[test["Embarked"] == "S"] = 0
test['Embarked'].loc[test['Embarked'] == 'C'] = 1
test['Embarked'].loc[test['Embarked'] == 'Q'] = 2

#Print the Sex and Embarked columns
print(train['Sex'])
print(train['Embarked'])


'''
step 3: create targets and features (feature engineering)
'''

#Create the target and features numpy arrays: target, features_one
target = train['Survived'].values

# Create a new array with the added features: features_two
features = train[["Pclass","Age","Sex","Fare", 'SibSp', 'Parch', 'Embarked']].values


'''
step 4: control overfitting and then training
'''

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree = my_tree.fit(features, target)

#Print the score of the new decison tree
print(my_tree.score(features, target))



#Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

#We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked']].values

#Building the Forest: my_forest
n_estimators = 100
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = n_estimators, random_state = 1)
my_forest = forest.fit(features_forest, target)

#Print the score of the random forest
print(my_forest.score(features_forest, target))


'''
step 5: prediction
'''


#Compute predictions and print the length of the prediction vector:test_features, pred_forest
test_features = test[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked']].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

#Request and print the `.feature_importances_` attribute
print(my_tree.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree.score(features, target))
print(my_forest.score(features_forest, target))


'''
step 6: save results
'''

# Make your prediction using the test set
my_prediction = my_forest.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test['PassengerId']).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])



