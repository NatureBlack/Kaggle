# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from time import time
import re
import argparse

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visual', action = 'store_true')
    parser.add_argument('-c', '--cross', action = 'store_true') 
    return parser.parse_args()
	
args = parse_args()
print(args)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

# step 1 : read in data

# open file
titanic_train = pd.read_csv("train.csv", dtype = {"Age": np.float64})
titanic_test = pd.read_csv("test.csv", dtype = {"Age": np.float64})

# merge the whole information from both train and test dataset
train_set = titanic_train.drop("Survived", axis = 1)
df_combo = pd.concat((train_set, titanic_test), axis = 0, ignore_index = True)

# fill in missing data
df_combo["Embarked"] = df_combo["Embarked"].fillna("C")



# step 2 : data preprocessing and cleaning

# Title and Surname Extraction
Title_list = pd.DataFrame(index = df_combo.index, columns = ["Title"])
Surname_list = pd.DataFrame(index = df_combo.index, columns = ["Surname"])

for (i, name) in enumerate(df_combo.Name) :
	parts = re.split('[,.] ', name)
	Surname_list.loc[i, 'Surname'] = parts[0]
	Title_list.loc[i, 'Title'] = parts[1]
	
# map the titles
Title_Dictionary = {
"Capt": "Officer",
"Col": "Officer",
"Major": "Officer",
"Jonkheer": "Sir",
"Don": "Sir",
"Sir" : "Sir",
"Dr": "Dr",
"Rev": "Rev",
"the Countess": "Lady",
"Dona": "Lady",
"Mme": "Mrs",
"Mlle": "Miss",
"Ms": "Mrs",
"Mr" : "Mr",
"Mrs" : "Mrs",
"Miss" : "Miss",
"Master" : "Master",
"Lady" : "Lady"
}    
    
def Title_Label(s):
    return Title_Dictionary[s]

df_combo["Title"] = Title_list["Title"].apply(Title_Label)
    
# find families
df_combo = pd.concat([df_combo, Surname_list], axis = 1)
df_combo['Fam'] = df_combo.Parch + df_combo.SibSp + 1

# family and title labels
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0

df_combo["Fam"] = df_combo.loc[:, "Fam"].apply(Fam_label)

def Title_label(s):
    if (s == "Sir") | (s == "Lady"):
        return "Royalty"
    elif (s == "Dr") | (s == "Officer") | (s == "Rev"):
        return "Officer"
    else:
        return s
        
df_combo["Title"] = df_combo.loc[:, "Title"].apply(Title_label)   

# cabin
Cabin_List = df_combo.loc[:, ["Cabin"]].fillna('UNK')
Cabin_Code = [cabin[0] for cabin in Cabin_List.Cabin]
Cabin_Code = pd.DataFrame({"Deck" : Cabin_Code})
df_combo = pd.concat([df_combo, Cabin_Code], axis = 1)

# tickets
def tix_clean(t):
	t = re.sub('[./ ]', '', t)
	return t
    
df_combo["Ticket"] = df_combo.loc[:, "Ticket"].apply(tix_clean)

# count unique tickets
Ticket_count = dict(df_combo.Ticket.value_counts())

# chang ticket into numbers
def Tix_ct(y):
    return Ticket_count[y]

df_combo["TicketGrp"] = df_combo.Ticket.apply(Tix_ct)

# ticket labels
def Tix_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

df_combo["TicketGrp"] = df_combo.loc[:, "TicketGrp"].apply(Tix_label) 

# drop unuse columns
df_combo.drop(["PassengerId", "Name", "Ticket", "Surname", "Cabin", "Parch", "SibSp"], axis=1, inplace = True)


# Filling missing Age data
# find unmissing index with other 3 columns
mask_Age = df_combo.Age.notnull()
Age_Sex_Title_Pclass = df_combo.loc[mask_Age, ["Age", "Title", "Sex", "Pclass"]]

# group by the other 3 columns and fillin with median value
Filler_Ages = Age_Sex_Title_Pclass.groupby(by = ["Title", "Pclass", "Sex"]).median()
Filler_Ages = Filler_Ages.Age.unstack(level = -1).unstack(level = -1)
print(Filler_Ages)

# find missing index with other 3 columns
mask_Age = df_combo.Age.isnull()
Age_Sex_Title_Pclass_missing = df_combo.loc[mask_Age, ["Title", "Sex", "Pclass"]]

def Age_filler(row):
	return Filler_Ages[row['Sex']].loc[row['Title'], row['Pclass']]
    
Age_Sex_Title_Pclass_missing["Age"]  = Age_Sex_Title_Pclass_missing.apply(Age_filler, axis = 1)   

df_combo["Age"] = pd.concat([Age_Sex_Title_Pclass["Age"], Age_Sex_Title_Pclass_missing["Age"]]) 

# fill in missing fare
dumdum = (df_combo.Embarked == "S") & (df_combo.Pclass == 3)
df_combo['Fare'].fillna(df_combo[dumdum].Fare.median(), inplace = True)


# OHE encoding nominal categorical features ###
df_combo = pd.get_dummies(df_combo)

# make final training and testing dataset
df_train = df_combo.loc[0 : len(titanic_train["Survived"]) - 1]
df_test = df_combo.loc[len(titanic_train["Survived"]) : ]
total_number_param = len(df_train.columns)
df_target = titanic_train.Survived

# visualization
if(args.visual) :
	matplotlib.style.use('ggplot')
	print(df_train.head(5))
	data = pd.crosstab(df_train.Age, df_target)
	data.plot()
	plt.show()

#check the constant feature
for i in range(total_number_param) :
	print('feature %d : ' % i, df_train.iloc[:, i].std())
	
# step 3 : training
# make training pipeline
select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')

pipeline = make_pipeline(select, clf)

# grid search for parameters with hyper parameters
if(args.cross) :
	pl = Pipeline([('sl', SelectKBest()), ('rf', RandomForestClassifier())])
	parameters = {'sl__k' : [15, 18, 20, 22, 25], 'rf__n_estimators' : [22, 24, 26, 28, 30], 'rf__max_depth' : [5, 6, 7, 8, 9, 10]}
	gs = GridSearchCV(pl, parameters, cv = 10)
	gs.fit(df_train, df_target)

	print("Best score: %0.3f" % gs.best_score_)
	best_parameters = gs.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

# train accuracy         
pipeline.fit(df_train, df_target)
predictions = pipeline.predict(df_train)
predict_proba = pipeline.predict_proba(df_train)[:, 1]
print("Accuracy : %.4g" % metrics.accuracy_score(df_target.values, predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(df_target, predict_proba))

# cross validation score
cv_score = cross_validation.cross_val_score(pipeline, df_train, df_target, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), 
np.min(cv_score),
np.max(cv_score)))

# classification score
(X_train, X_test, y_train, y_test) = cross_validation.train_test_split(df_train, df_target, test_size = 0.2, random_state = 20)
y_pred = pipeline.predict(X_test)
target_names = ['survive', 'die']
labels = [1, 0]
print(metrics.classification_report(y_test, y_pred, target_names = target_names, labels = labels))

# prediction
final_pred = pipeline.predict(df_test)

# submission
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": final_pred })
submission.to_csv("yzp.csv", index = False) 