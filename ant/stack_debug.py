import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import Imputer, StandardScaler, Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from lib.model_performance import *
from lib.data_processing import *

param = {
        "objective" : "binary:logistic",
        "max_depth" : 4,
        "subsample" : 0.8,
        "colsample_bytree" : 0.8,
        "min_child_weight" : 1,
        "gamma" : 0,
        "eta" : 0.07, #learning_rate
        "eval_metric" : ['error'], #early stop only effects on error
        "silent" : 0
        }
num_round = 1

path1 = os.path.abspath(".")

#***********************model & score path *************************************#
#model save path#
model_path = path1 + "/save_restore/"
#where to save the figure & answer & hParams
score_path = path1 + "/score/"

#***********************data_path**********************************************#
data_path = path1 + "/data/"
#train_path = "/home/lecui/kaggle/data/train.npy"
train_path = data_path + "train.npy"
#test_data path
test_path = data_path + "test_a.npy"

stack_test_path = score_path + "stack_test_sheet.csv"

now = datetime.datetime.now()
log_file = "log/"+ "{}_".format(now.year)+"{}_".format(now.month)+"{}/".format(now.day)
final_test_path = log_file + "score_{}:{}.csv".format(now.hour, now.minute)

fmap = path1 + "/fmap/xgb.fmap"

def progress_log(names = None, classifiers = None, name = None, end_log = False, start_log = False):
    #now = datetime.datetime.now()
    #log_file = "log/"+ "{}_".format(now.year)+"{}_".format(now.month)+"{}/".format(now.day)
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    else:
        if end_log:
            with open(log_file + "{}_".format(now.year)+"{}_".format(now.month)+"{}".format(now.day), "a") as f:
                f.write(str('>'*40)+  str("END LOG") + str('<'*40) + '\n')
                f.write(str('>'*40)+  str("{}:{}".format(now.hour, now.minute)) + str('<'*40) + '\n')
                f.write('\n')
                f.write('\n')
        elif start_log:
            with open(log_file + "{}_".format(now.year)+"{}_".format(now.month)+"{}".format(now.day), "a") as f:
                f.write(str('>'*40)+  str("Start LOG") + str('<'*40) + '\n')
                f.write(str('>'*40)+  str("{}:{}".format(now.hour, now.minute)) + str('<'*40) + '\n')
                f.write('\n')
        else:
            with open(log_file + "{}_".format(now.year)+"{}_".format(now.month)+"{}".format(now.day), "a") as f:
                f.write(str('#'*35)+  str(name) + str('#'*35) + '\n')
                f.write('\n')
            for n, clf in zip(names, classifiers):
                with open(log_file + "{}_".format(now.year)+"{}_".format(now.month)+"{}".format(now.day), "a") as f:
                    f.write(str('*'*30) + str(n) + str('*'*30) + '\n')
                    f.write(str(clf) + "\n")

# ####################Feature Processing####################
# Including 1. Imputation,
#           2. Standardization,
#           3. Normalizaiton
# ##########################################################

def feature_processing(names,preprocessors,feature,test_feature):
    progress_log(names, preprocessors)
    for name, preprocessor in zip(names,preprocessors):
        print("Start {}".format(name))
        preprocessor.fit(feature)
        feature = preprocessor.transform(feature)
        test_feature = preprocessor.transform(test_feature)
    return feature, test_feature

# ####################Feature Engineer######################
def select_feature_from_xgb(feature,labels,test_feature):

    print("\nStart selecting importance feature")
    xgb = XGBClassifier(n_estimators=2, max_depth=4, learning_rate = 0.07, subsample = 0.8, colsample_bytree = 0.9)
    xgb = xgb.fit(feature, labels)
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]

    model = SelectFromModel(xgb, prefit=True)
    feature_new = model.transform(feature)
    test_feature_new = model.transform(test_feature)
    with open(data_path + "importance_feature.txt" , "w") as log:
        for f in range(feature_new.shape[1]):
            log.write(str(f + 1) + "." +  " feature " +  str(indices[f]) + "  " + str(importances[indices[f]]) + "\n")
            #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print("feature selection done saved new data in data path")

    return feature_new, test_feature_new

# ####################CV Slicing############################
def stack_split(feature, labels, number_of_model):
    # Define number of sizes per model
    fold_size = int(labels.size/number_of_model)

    # Iterate number of models to get different fold, feature and label data
    fold_split = {}
    feature_split = {}
    label_split = {}
    fold_split_label = {}

    for i in range(number_of_model):
        # define starting and end rows of the fold data
        start_row = fold_size * i
        end_row = fold_size * (i+1)

        if i == number_of_model - 1:

            print("\nfold_{}".format(i+1) + " starting between row:{}".format(start_row) + " and row:{}".format(end_row))
            fold_split["fold_{}".format(i+1)] = feature[start_row:,:]
            fold_split_label["fold_label_{}".format(i+1)] = labels[start_row:]
            # Delete the extrated data from feature and label data
            feature_split["feature_{}".format(i+1)] = np.delete(feature, np.s_[start_row:], axis = 0)
            label_split["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:], axis = 0)

        else:


            print("\nfold_{}".format(i+1) + " starting between row:{}".format(start_row) + " and row:{}".format(end_row))
            # Store extrated fold data from feature
            fold_split["fold_{}".format(i+1)] = feature[start_row:end_row,:]
            fold_split_label["fold_label_{}".format(i+1)] = labels[start_row:end_row]
            # Delete the extrated data from feature and label data
            feature_split["feature_{}".format(i+1)] = np.delete(feature, np.s_[start_row:(start_row + fold_size)], axis = 0)
            label_split["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:(start_row + fold_size)], axis = 0)

    return fold_split, fold_split_label, feature_split, label_split

def stack_xgb(train_path, label, test_path):

    dtrain = xgb.DMatrix(train_path, label=label)
    dtest = xgb.DMatrix(test_path)
    bst = xgb.train(param, dtrain, num_round)
    final_preds = bst.predict(dtest)

    return final_preds

def stack_layer(names, classifiers, feature, labels, test_feature, layer_name):

        progress_log(names, classifiers, layer_name)
        fold_split, fold_split_label, feature_split, label_split = stack_split(feature,labels,5)
        layer_transform_train = []
        layer_transform_test = []
        weighted_avg_roc = []
        for name, clf in zip(names, classifiers):
            fold_score = []
            test_score = []
            roc_list = []

            for i in range(len(fold_split)):
                start = time.time()
                print("\nProcessing model :{} fold {}".format(name, i+1))
                clf.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
                print("Training complete")
                stack_score = clf.predict_proba(fold_split["fold_{}".format(i+1)])
                print("fold score predicted")
                test_prediction = clf.predict_proba(test_feature)
                print("test score predicted")
                test_score.append(test_prediction[:,1].tolist())
                fold_score += stack_score[:,1].tolist()
                print("model {}".format(name) + " complete")
                roc = offline_model_performance(fold_split_label["fold_label_{}".format(i+1)], stack_score[:,1])
                roc_list.append(roc)
                print("\n# Fold {} performace is {:4f}".format(i+1, roc))
                end = time.time()
                print(">>>>Duration<<<< : {}min ".format(round((end-start)/60,2)))

            roc = np.array(roc_list)
            roc_mean = np.mean(roc, axis = 0)
            roc_std = np.std(roc, axis = 0)
            print("##"*40)
            print("\n# ROC_1(JL) :{} (+/- {:2f})".format(roc_mean, roc_std*2))
            print("##"*40)
            #Averaging stacked
            stack_test_layer1_preds = np.stack(test_score, 1)
            #averaging stacked data
            avged_test_preds = []

            for row in stack_test_layer1_preds:
                avg = np.mean(row)
                avged_test_preds.append(avg)

            print("\nAveraging test score done ......")

            weighted_avg_roc.append(roc_mean)
            layer_transform_train.append(fold_score)
            layer_transform_test.append(avged_test_preds)

        weighted_avg_roc = np.array(weighted_avg_roc).transpose()
        layer_transform_train = np.array(layer_transform_train).transpose()
        layer_transform_test = np.array(layer_transform_test).transpose()
        np.savetxt(log_file + "{}_train_{}:{}.csv".format(layer_name, now.hour, now.minute) ,layer_transform_train , fmt = '%.9f', delimiter = ',')
        np.savetxt(log_file + "{}_test_{}:{}.csv".format(layer_name, now.hour, now.minute) ,layer_transform_test ,fmt = '%.9f', delimiter = ',')
        return layer_transform_train, layer_transform_test, weighted_avg_roc

def two_layer_stacking(train_data, test):

    train_data = train_data.values
    test = test.values
    feature = train_data[:,3:]
    label = train_data[:,1].astype(int)
    test = test[:,2:]

    # ####################First Layer Start#####################
    clf_names = ["XGB", "RF", "LR", "ET", "GBDT"]
    classifier = [
            XGBClassifier(n_estimators=30, max_depth=3, learning_rate = 0.06, #380
                    gamma = 0.1, n_jobs = -1, subsample = 0.8, colsample_bytree = 0.8),
            RandomForestClassifier(n_estimators = 30, min_samples_split = 110, max_depth = 25, criterion='entropy', n_jobs = -1), #160
            #LogisticRegression(class_weight = "balanced", C = 2),
            ExtraTreesClassifier(n_estimators = 30, n_jobs = -1, min_samples_split = 70),
            GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.08, n_jobs = -1),
    ]
    print(classifier)
    feature, test = stack_layer(clf_names, classifier, feature, label, test, layer_name = "layer1")

    layer2_clf_names = ["XGB", "RF"]

    layer2_classifier = [
        XGBClassifier(n_estimators=30, max_depth=4, learning_rate = 0.07,
                          gamma = 0, n_jobs = -1,
                          subsample = 0.8, colsample_bytree = 0.8),

        RandomForestClassifier(n_estimators = 30, min_samples_split = 110, max_depth = 20, criterion='entropy', n_jobs = -1),
    ]
    print(layer2_classifier)

    feature, test, avg_roc = stack_layer(layer2_clf_names, layer2_classifier, feature, label, test, layer_name = "layer2")

    avg_roc = np.average(avg_roc, axis =1, weights=[3./4, 1./4])
    print("\n# Average 5-FOLD ROC : ".format(avg_roc))

    final_preds = np.average(test, axis =1, weights=[3./4, 1./4])

    #final_preds = stack_xgb(feature, label, test)

    return final_preds

#
#     # ####################First Layer Start#####################
#     clf_names = ["XGB", "RF", "MLP", "LR"]
#     classifier = [
#
#         XGBClassifier(n_estimators=450, max_depth=4, learning_rate = 0.02,
#                       gamma = 0.2, reg_alpha = 0.07,
#                       subsample = 0.6, colsample_bytree = 0.7),
#
#         RandomForestClassifier(n_estimators = 450, max_depth = 4, criterion='entropy'), #450
#         MLPClassifier(hidden_layer_sizes=(256,128,128), activation = "logistic", batch_size = 20000)
#         #LogisticRegression(class_weight = "balanced"),
#
#     ]
#
#     feature, test = stack_layer(clf_names, classifier, feature, label, test, layer_name = "layer1")
#
#     # ####################Second Layer Start#####################
#     layer2_clf_names = ["XGB", "KNN", "QDA"]
#
#     layer2_classifier = [
#         XGBClassifier(n_estimators=450, max_depth=4, learning_rate = 0.02,
#                           gamma = 0.2, reg_alpha = 0.07,
#                           subsample = 0.6, colsample_bytree = 0.7),
#         KNeighborsClassifier(n_neighbors=5, weights='uniform', leaf_size=30, n_jobs=-1),
#         QDA(),
#         #ExtraTreesClassifier(n_estimators = 450, max_depth = 4, criterion='entropy'),
#         #LogisticRegression(class_weight = "balanced"),
#         #MLPClassifier(hidden_layer_sizes=(256,128,128), activation = "logistic", batch_size = 20000)
#         #RandomForestClassifier(n_estimators = 3, max_depth = 4, criterion='entropy'), #450
#     ]
#
#
