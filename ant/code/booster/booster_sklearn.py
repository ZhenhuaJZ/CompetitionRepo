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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import operator
import warnings
import sys
sys.path.insert(0,"/home/stirfryrabbit/Documents/CompetitionRepo/ant/code/preprocessing")
from preprocessing import replace_missing_by_custom_mode

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'


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


def save_score(preds, param_name):
    now = datetime.datetime.now()

    answer_sheet = pd.read_csv("../../tool/answer_sheet.csv")
    #Dataframe data
    answer_sheet = pd.DataFrame(answer_sheet)
    #Feed result in score column
    answer = answer_sheet.assign(score_1 = preds[:,0])
    answer = answer.assign(score_2 = preds[:,1])
    #Save to .csv
    score_path = "score/{}_{}_{}:{}/".format(now.month, now.day, now.hour, now.minute)
    os.makedirs(score_path)
    answer.to_csv(score_path + "{}_score_{}d{}m{}h.csv".format(param_name, now.day, now.month, now.hour), index = None, float_format = "%.9f")

    return print("Score saved in {}".format(score_path))

def main():
    train_mode_path = "../../data/train.csv" #train_heatmap , train_mode_fill_leo, train, train_mode_fill_full_feat
    test_mode_path = "../../data/test_a.csv" #test_a_heatmap, test_a_mode_fill, test_a, test_mode_fill_full_feat
    train_data = pd.read_csv(train_mode_path)
    train_data = train_data[(train_mode_data.label==0)|(train_mode_data.label==1)]
    test_data = pd.read_csv(test_mode_path)
    print("\nreaded data")
    # Fill NaN data with mode
    train_data, test_data = replace_missing_by_custom_mode(train_data,test_data)
    # Extract features and labels from train and test data set
    feature = train_data.iloc[:,3:]
    label = train_data.iloc[:,1]
    test = test_data.iloc[:,2:]
    # Delete the original data to save memory
    del test_data, train_data
    # Using XGB for classifier`
    print("Start selecting importance features")
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate = 0.07,
    	                subsample = 0.8, colsample_bytree = 0.9, n_jobs = -1)
    exit()
    print("Initialised classifiers")
    xgb = xgb.fit(feature, label,
                  # eval_set = [(train_mode_test, label_mode_test)],
                  # eval_metric = "auc",
                  # verbose = True
                  )
    score_mode = xgb.predict_proba(test)
    print(score_mode)
    save_score(score_mode,"mode")

if __name__ == "__main__":
    main()
