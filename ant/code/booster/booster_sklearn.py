"""
@Authors Leo.cui
7/5/2018
Xgboost
"""
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
import operator
import warnings

train_path = "../../data/train_mode_fill.csv" #train_heatmap , train_mode_fill, train,
test_path = "../../data/test_a_mode_fill.csv" #test_a_heatmap, test_a_mode_fill, test_a,

train_data = pd.read_csv(train_path)
train_data = train_data[(train_data.label==0)|(train_data.label==1)]
test = pd.read_csv(test_path)
#Define training set
train = train_data.iloc[:,3:]
label = train_data.iloc[:,1]
test = test.iloc[:,2:]

print("\nStart selecting importance features")
xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate = 0.07, 
	                subsample = 0.7, colsample_bytree = 0.9, n_jobs = -1)

xgb = xgb.fit(train, label)
score1 = xgb.predict_proba(test)
print(score1)