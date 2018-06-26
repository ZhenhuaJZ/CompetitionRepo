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

params_path = "model_params/{}_{}_{}:{}/".format(now.month, now.day, now.hour, now.minute)
score_path = "score/{}_{}_{}:{}/".format(now.month, now.day, now.hour, now.minute)
as_path = "../../tool/answer_sheet.csv"

#Pd read path
train_data = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_data = train_data[(train_data.label==0)|(train_data.label==1)]
#Main data
train = train_data.iloc[3:200,3:]
labels = train_data.iloc[3:200,1]
test = test.iloc[3:200,2:]

def custom_imputation(df_train, df_test, fillna_value = None):
    train = df_train.fillna(fillna_value)
    test = df_test.fillna(fillna_value)
    return train, test

def save_score(preds):
    low_risk = 0
    medium_risk = 0
    high_risk = 0
    for p in preds:
        if p > 0.2 and p < 0.4:
            low_risk += 1

        elif p >= 0.4 and p < 0.6:
            medium_risk += 1

        elif p >= 0.6:
            high_risk += 1

    print("probability [0.2 ~ 0.4] rate : {:.3f}%\n".format(100*(low_risk/len(preds))),
          "probability [0.4 ~ 0.6] rate : {:.3f}%\n".format(100*(medium_risk/len(preds))),
          "probability [0.6 ~ 1.0] rate : {:.3f}%\n".format(100*(high_risk/len(preds))))

    answer_sheet = pd.read_csv(as_path)
    answer_sheet = pd.DataFrame(answer_sheet)
    answer = answer_sheet.assign(score = preds)
    answer.to_csv(score_path + "{}_score_{}d{}m{}h.csv".format(param_name, now.day, now.month, now.hour), index = None, float_format = "%.9f")
    return print("Score saved in {}".format(score_path))

def main():
    start = time.time()
    os.makedirs(score_path)
    train, test = custom_imputation(train, test, fillna_value = 1)
    #split data
    #train, validation, label, validation_label = train_test_split(train, labels, test_size = 0.3, random_state = 42)
    #feature preprocessing
    imputer = Imputer(missing_values='NaN', strategy='mean') #mean ,median, most_frequent
    standar = StandardScaler(with_mean=True, with_std=True)
    norm = Normalizer(norm = 'l2') #norm l1, l2
    #feature selection
    tbfs = XGBClassifier(max_depth = 3, n_estimators = 400, subsample = 0.8,
                        colsample_bytree = 0.8, learning_rate = 0.08, n_jobs = -1) #tree base feature_selection
    kbest = SelectKBest(chi2)
    #clf
    xgb = XGBClassifier(max_depth = 3, n_estimators = 400, subsample = 0.9,
                        colsample_bytree = 0.8, learning_rate = 0.1, n_jobs = -1)
    rdforest = RandomForestClassifier(n_jobs = -1)
    grdboost = GradientBoostingClassifier(n_jobs = -1)

    # ###########################Tuning Params################################

    params = [
              [{"tbfs__min_child_weight" : [1,2,3]}, {"xgb__max_depth" : [3, 4], "xgb__min_child_weight" : [1, 2, 3]}],
              [{"tbfs__subsample" : [0.9, 0.8]}, {"xgb__gamma" : [0.1, 0.2], "xgb__subsample" : [0.9, 0.8, 0.7], "xgb__colsample_bytree" : [0.9, 0.8, 0.7]}],
              [{"tbfs__colsample_bytree" : [0.8, 0.7]}, {"xgb__reg_alpha" : [0.01, 0.05, 0.06], "xgb__scale_pos_weight" : [1, 10, 20]}],
              [{"xgb__learning_rate" : [i*0.01 for i in range(3,9)]}]
             ]

    pipe = Pipeline(steps = [ #('imputer', Imputer()),
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
    bst_params = clf.best_params_
    bst_score = clf.best_score_
    bst_estimator = clf.best_estimator_
    print("\nBest parameters set found on development set: + \n +{}".format(bst_params))
    probs = clf.predict_proba(test)
    #save score
    save_score(probs[1])

    # ###############################save params################################

    with open(params_path  + "params.txt", 'a') as f:
        f.write(
                "************************" + "\n"
                + str(bst_estimator) + "\n"
                + str(bst_params) + "\n"
                + str(bst_score))

    print("Find best params {}, with best roc {}".format(bst_params, bst_score))
    plot.grid_search(clf.grid_scores_, change='max_depth', kind ='bar')
    plt.savefig(params_path + "grid_params_{}.png".format(round(bst_score,2)))
    end = time.time()
    print(">>>>Duration<<<< : {}min ".format(round((end-start)/60,2)))

if __name__ == '__main__':
    warnings.filterwarnings(module = 'sklearn*',
                            action = 'ignore', category = DeprecationWarning)
    main()
