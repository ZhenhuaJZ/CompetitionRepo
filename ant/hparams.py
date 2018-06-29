import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import datetime, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, Normalizer, MaxAbsScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from xgboost import XGBClassifier
from  sklearn.ensemble  import  GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.grid_search import GridSearchCV
from tempfile import mkdtemp
from shutil import rmtree
cachedir = mkdtemp()

# #####################Feature Preprocessing#############################
imputer = Imputer(missing_values='NaN', strategy='mean') #mean ,median, most_frequent
standar = StandardScaler(with_mean=True, with_std=True)
maxabs_std = MaxAbsScaler()
minmax_std = MinMaxScaler()
norm = Normalizer(norm = 'l2') #norm l1, l2
# #####################Feature Selection#################################
kbest = SelectKBest(chi2)
# #####################Feature Reduction#################################

# #####################Classcifiers######################################
xgb = XGBClassifier(max_depth = 4, n_estimators = 460, subsample = 0.6,min_child_weight = 2,gamma = 0,
					colsample_bytree = 0.9, learning_rate = 0.1)
rf = RandomForestClassifier(n_estimators = 1, criterion = "entropy", max_depth = 13,
		 			min_samples_split = 110, min_samples_leaf = 1, max_leaf_nodes = None)

method_1_describ = ["MinMaxScaler", "Kbest", "Xgboost"]
method_2_describ = ["StandardScaler", "Tree-Base Importance Feature", "Xgboost"]
method_3_describ = ["Xgboost"]
method_4_describ = ["rf"]

# ###########################Tuning Params################################
params_1 = [
          #[{
		   #"kbest__k" : [80,100,120],
           #"xgb__max_depth" : [3, 4],
           #"xgb__min_child_weight" : [1, 2],
          #}],
          [{
           "xgb__gamma" : [0, 0.1],
           "xgb__subsample" : [0.6, 0.5],
           "xgb__colsample_bytree" : [0.9, 0.8],
          }],

          [{
          #"tbfs__colsample_bytree" : [0.8, 0.7],
          "xgb__reg_alpha" : [0.05, 0.07],
          "xgb__scale_pos_weight" : [20, 30],
          }],

          [{
          "xgb__learning_rate" : [i*0.01 for i in range(3,8)]
          }]

         ]
params_2 = [
          [{
           "xgb__max_depth" : [3,4],
           "xgb__min_child_weight" : [1, 2],
          }],

          [{
           "xgb__gamma" : [0, 0.1],
		   "xgb__n_estimators" : [3,4,5],
          }],

          [{
		  "xgb__scale_pos_weight" : [20, 30],
		  "xgb__subsample" : [0.6, 0.5],
		  "xgb__colsample_bytree" : [0.9, 0.8],
          }],

          [{
		  "xgb__learning_rate" : [i*0.01 for i in range(3,8)],
		  "xgb__reg_alpha" : [0.05, 0.07],
          }]
         ]

params_3 = [
          #[{
           #"xgb__max_depth" : [3,4],
           #"xgb__min_child_weight" : [1, 2],
          #}],

          #[{
           #"xgb__gamma" : [0, 0.1],
		   #"xgb__scale_pos_weight" : [1, 20],
		   #"xgb__n_estimators" : [3,4,5],
          #}],

          #[{
		  #"xgb__subsample" : [0.7, 0.6], #try [0.6,0.5]
		  #"xgb__colsample_bytree" : [0.9, 0.8],
          #}],

          [{
		  "xgb__learning_rate" : [i*0.01 for i in range(6,8)],
		  "xgb__reg_alpha" : [0.05, 0.07],
          }]
         ]
params_4 = [
          [{
           "rf__max_depth" : [17, 20, 25],
           "rf__n_estimators" : [260,330],
          }],

          #[{
           #"xgb__gamma" : [0, 0.1],
		   #"xgb__scale_pos_weight" : [1, 20],
		   #"xgb__n_estimators" : [3,4,5],
          #}],

         ]
# ##########################PipeLine#####################################
pipe_family = {
			   "pipe_1" : Pipeline([('minmax_std', minmax_std), ('kbest', kbest), ('xgb', xgb)], memory = cachedir),
			   "pipe_2" : Pipeline([('standar', standar), ('tbfs', SelectFromModel(xgb)), ('xgb', xgb)], memory = cachedir),
			   "pipe_3" : Pipeline([('xgb', xgb)], memory = cachedir),
			   "pipe_4" : Pipeline([('rf', rf)], memory = cachedir),
			   }

strategy = {
			"method_1": (params_1, pipe_family["pipe_1"], method_1_describ),
			"method_2": (params_2, pipe_family["pipe_2"], method_2_describ),
			"method_3": (params_3, pipe_family["pipe_3"], method_3_describ),
			"method_4": (params_4, pipe_family["pipe_4"], method_4_describ),
			}
