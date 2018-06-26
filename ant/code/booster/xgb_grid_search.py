import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn_evaluation import plot
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
import datetime, time
now = datetime.datetime.now()

#where to save the figure & answer & hParams
params_path = "model_params/{}_{}_{}:{}/".format(now.month, now.day, now.hour, now.minute)
#***********************data_path**********************************************#
#train_data path
train_path = "../../data/train_heatmap.csv" #train_heatmap , train_mode_fill, train,
#test_data path
test_path = "../../data/test_a_heatmap.csv" #test_a_heatmap, test_a_mode_fill, test_a,

#Pd read path
train_data = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_data = train_data[(train_data.label==0)|(train_data.label==1)]
#Main data
features = train_data.iloc[1:1000,3:]
labels = train_data.iloc[1:1000,1]
test_feature = test.iloc[1:1000,2:]


def grid_search_xgb(features,labels,test_feature):
    os.makedirs(params_path)

    tuning_params = {
                        "max_depth" : [3,4], #[3,4,5,6]
                        "min_child_weight" : [1,2],
                        "n_estimators" : [450,460,480], #[350,400,450,480,500]
                        "gamma" : [0,0.1,0.2], # [0,0.1,0.2]
                        "subsample" : [0.7,0.6], # [0.9,0.8,0.7]
                        "colsample_bytree" : [0.9,0.8,0.7], # [0.9,0.8,0.7]
                        "reg_alpha" : [0.01, 0.05, 0.06, 0.07]
                        "scale_pos_weight" : [1,10,20]
                        "learning_rate" : [i*0.01 for i in range(10)]
                        }

    score = ['roc_auc']

    parameter_space = {}

    start = time.time()
    print("# Tuning hyper-parameters for %s" % score)
    print("\nProcessing XGB model")
    #clf = XGBClassifier(objective = "binary:logistic", n_estimators = 450, max_depth =4, gamma = 0,reg_alpha =0,
    #					subsample = 0.8 , min_child_weight =1, colsample_bytree = 0.8, learning_rate = 0.1,
    #					scale_pos_weight = 1, n_jobs = -1
    # 					)
    #clf = XGBClassifier(objective = "binary:logistic", n_jobs = -1, kwargs = parameter_space)

    clf = GridSearchCV(pipe, parameter_space, cv= 5, scoring='%s' % score)
    clf.fit(features, labels)
    print("Best parameters set found on development set:")
    bst_params = clf.best_params_
    bst_score = clf.best_score_
    bst_estimator = clf.best_estimator_

    with open(params_path  + "XGB_params.txt", 'a') as f:
        f.write(
                "************************" + "\n"
                + str(bst_estimator) + "\n"
                + str(bst_params) + "\n"
                + str(bst_score))

    print("Find best params {}, with best roc {}".format(bst_params, bst_score))
    print("XGB model complete")
    plot.grid_search(clf.grid_scores_, change='max_depth', kind ='bar')
    plt.savefig(params_path + "XGB_params_{}.png".format(round(bst_score,2)))
    end = time.time()
    print(">>>>Duration<<<< : {}min ".format(round((end-start)/60,2)))

    return bst_estimator


def main():

    clf = grid_search_xgb(features,labels,test_feature)

if __name__ == '__main__':
    main()
