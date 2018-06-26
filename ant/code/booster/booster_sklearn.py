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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def select_features_from_xgb(features,labels,test_feature):
    print("\nStart selecting importance features")
    xgb = XGBClassifier(n_estimators=2, max_depth=4, learning_rate = 0.07, subsample = 0.8, colsample_bytree = 0.9)
    xgb = xgb.fit(features, labels)
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]

    model = SelectFromModel(xgb, prefit=True)
    features_new = model.transform(features)
    test_feature_new = model.transform(test_feature)
    with open(data_path + "importance_features.txt" , "w") as log:
        for f in range(features_new.shape[1]):
            log.write(str(f + 1) + "." +  " feature " +  str(indices[f]) + "  " + str(importances[indices[f]]) + "\n")
            #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print("Features selection done saved new data in data path")
    return features_new, test_feature_new

def select_features_from_univariate(features,labels,test_feature, n_of_feature):
    print("\nStart selecting feature with RFE")
    select_feature = SelectKbest(chi2, k = n_of_feature).fit(features,labels)
    features_new = select_feature.transform(features)
    test_feature_new = select_feature.transform(test_feature)
    print("End of RFE feature select")
    return features_new, test_feature_new

def feature_selection(names, classifiers, feature, label, test_feature):
    for name , clf in zip(names,classifiers):
        select_feature = clf.fit(feature,label)
        importances = xgb.feature_importances_
        indices = np.argsort(importances)[::-1]

        feature_new = select_feature.transform(feature)
        test_feature_new = select_feature.transform(test_feature)

        with open(data_path + "importance_features.txt" , "w") as log:
            log.write(name)
            for f in range(features_new.shape[1]):
                log.write(str(f + 1) + "." +  " feature " +  str(indices[f]) + "  " + str(importances[indices[f]]) + "\n")
                #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        print("Features selection done saved new data in data path")
    return feature_new, test_feature_new

def feature_selection(feature,label,test_feature,select_mode = "xgb",n_of_feature = "100"):
    if select_mode == "xgb":
        feature_new, test_feature_new = select_features_from_xgb(feature,label,test_feature,n_of_feature)
    elif select_mode == "univariate":
        feature_new, test_feature_new = select_features_from_univariate(feature,label,test_feature,n_of_feature)
    return feature_new, test_feature_new


train_mode_path = "../../data/train_mode_fill.csv" #train_heatmap , train_mode_fill, train,
test_mode_path = "../../data/test_a_mode_fill.csv" #test_a_heatmap, test_a_mode_fill, test_a,

train_heatmap_path = "../../data/train_heatmap.csv"
test_heatmap_path = "../../data/test_a_heatmap.csv"

train_heatmap_data = pd.read_csv(train_heatmap_path)
train_heatmap_data = train_heatmap_data[(train_heatmap_data.label==0)|(train_heatmap_data.label==1)]
test_heatmap_data = pd.read_csv(test_heatmap_path)
print("Read and processed heatmap data")

#Define training set
train_heatmap = train_heatmap_data.iloc[:,3:]
label_heatmap = train_heatmap_data.iloc[:,1]
test_heatmap = test_heatmap_data.iloc[:,2:]

train_mode_data = pd.read_csv(train_mode_path)
train_mode_data = train_mode_data[(train_mode_data.label==0)|(train_mode_data.label==1)]
test_mode_data = pd.read_csv(test_mode_path)

#Define training set
train_mode = train_mode_data.iloc[:,3:]
label_mode = train_mode_data.iloc[:,1]
test_mode = test_mode_data.iloc[:,2:]
print("Read and processed mode data")


print("\nStart selecting importance features")
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate = 0.07,
	                subsample = 0.7, colsample_bytree = 0.9, n_jobs = -1)

xgb = xgb.fit(train_mode, label_mode)
score_mode = xgb.predict_proba(test_mode)

print("predicted mode data")
print(score_mode)

xgb = xgb.fit(train_heatmap, label_heatmap)
score_heatmap = xgb.predict_proba(test_heatmap)

print("Predicted heatmap data")
print(score_heatmap)

score_mode.to_csv("mode.csv")
score_heatmap.to_csv("heatmap.csv")
