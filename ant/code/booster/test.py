import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.feature_selection import chi2
from sklearn_evaluation import plot
from sklearn.grid_search import GridSearchCV
import operator
import warnings

#train_data path
train_path = "../../data/train_fill_0.csv" #train_heatmap , train_mode_fill, train,
#test_data path
test_path = "../../data/test_a_heatmap.csv" #test_a_heatmap, test_a_mode_fill, test_a,

#Pd read path
train_data = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_data = train_data[(train_data.label==0)|(train_data.label==1)]
#Main data
features = train_data.iloc[3:1000,3:]
labels = train_data.iloc[3:1000,1]
test_feature = test.iloc[3:1000,2:]


warnings.filterwarnings(module = 'sklearn*', action = 'ignore', category = DeprecationWarning)
feats = XGBClassifier()
kbest = SelectKBest(chi2)
xgb = XGBClassifier()

#xgb_feats_select = xgb_feats_select.fit(features, labels)

params = [{"xgb__n_estimators" : [10, 12]}, 
          {"xgb__n_estimators" : [14, 14]}]

pipe = Pipeline(steps = [('feats', SelectFromModel(feats)), 
	                     ('xgb', xgb)])

for p in params:
	print(p)
	clf = GridSearchCV(pipe, p)
	print(clf)

#print("Best parameters set found on development set:")
#bst_params = estimator.best_params_
#bst_score = estimator.best_score_
#bst_estimator = estimator.best_estimator_
#print(bst_score)
_clf = clf.fit(features, labels)
#probs = _clf.predict_proba(test_feature)
#print(probs[1])