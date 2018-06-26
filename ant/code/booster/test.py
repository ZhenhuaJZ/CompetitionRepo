import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import datetime, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, Normalizer
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from xgboost import XGBClassifier
from  sklearn.ensemble  import  GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn_evaluation import plot
import operator
import warnings

#train_data path
train_path = "../../data/train.csv" #train_heatmap , train_mode_fill, train,
#test_data path
test_path = "../../data/test_a.csv" #test_a_heatmap, test_a_mode_fill, test_a,

#Pd read path
train_data = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_data = train_data[(train_data.label==0)|(train_data.label==1)]
#Main data
train = train_data.iloc[3:1000,3:]
labels = train_data.iloc[3:1000,1]
test = test.iloc[3:300,2:]

def custom_imputation(df_train, df_test, fillna_value = None):
    train = df_train.fillna(fillna_value)
    test = df_test.fillna(fillna_value)
    return train, test




def main():

    #train, test = custom_imputation(train, test, fillna_value = 0)
    #split data
    #train, validation, label, validation_label = train_test_split(train, labels, test_size = 0.3, random_state = 42)
    #feature preprocessing
    imputer = Imputer(missing_values='NaN', strategy='mean') #mean ,median, most_frequent
    standar = StandardScaler(with_mean=True, with_std=True)
    norm = Normalizer(norm = 'l2') #norm l1, l2
    #feature selection
    tbfs = XGBClassifier() #tree base feature_selection
    kbest = SelectKBest(chi2)
    #clf
    xgb = XGBClassifier(n_jobs = -1)
    rdforest = RandomForestClassifier()
    grdboost = GradientBoostingClassifier()

    params = [{"tbfs__n_estimators" : [10, 12]},
              {"xgb__n_estimators" : [14, 14]}]

    pipe = Pipeline(steps = [('imputer', Imputer()),
                               ('tbfs', SelectFromModel(tbfs)),
    	                       ('xgb', xgb)])

    #pipe_2 = Pipeline(steps = [('imputer', Imputer()),
                               #('kbest', kbest),
    	                       #('xgb', xgb)])

    #pipe_3 = Pipeline(steps = [('imputer', Imputer()),
                               #('tbfs', SelectFromModel(tbfs)),
    	                       #('xgb', xgb)])

    for param in params:
    	clf = GridSearchCV(pipe, param_grid  = param, scoring = 'roc_auc',
                           verbose = 1, n_jobs = -1, cv = 5)



    clf = clf.fit(train, labels)

    print("\nBest parameters set found on development set:")
    bst_params = clf.best_params_
    bst_score = clf.best_score_
    bst_estimator = clf.best_estimator_
    print(bst_params)
    print(bst_score)
    probs = bst_estimator.predict_proba(test)
    #print(probs)

if __name__ == '__main__':
    warnings.filterwarnings(module = 'sklearn*', action = 'ignore', category = DeprecationWarning)
    main()
