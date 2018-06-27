import xgboost as xgb
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
from sklearn_evaluation import plot
import operator
import warnings
now = datetime.datetime.now()

#train_data path
train_path = "../../data/train.csv" #train_heatmap , train_mode_fill, train,
#test_data path
test_path = "../../data/test_a.csv" #test_a_heatmap, test_a_mode_fill, test_a,

params_path = "model_params/{}_{}_{}:{}/".format(now.month, now.day, now.hour, now.minute)
score_path = "score/{}_{}_{}:{}/".format(now.month, now.day, now.hour, now.minute)
as_path = "../../tool/answer_sheet.csv"

#Pd read path
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
train_data = train_data[(train_data.label==0)|(train_data.label==1)]
#Main data
train = train_data.iloc[:,3:]
labels = train_data.iloc[:,1]
test = test_data.iloc[:,2:]

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
	"""
    print("probability [0.2 ~ 0.4] rate : {:.3f}%\n".format(100*(low_risk/len(preds))),
          "probability [0.4 ~ 0.6] rate : {:.3f}%\n".format(100*(medium_risk/len(preds))),
          "probability [0.6 ~ 1.0] rate : {:.3f}%\n".format(100*(high_risk/len(preds))))
	"""
	answer_sheet = pd.read_csv(as_path)
	answer_sheet = pd.DataFrame(answer_sheet)
	answer = answer_sheet.assign(score = preds)
	answer.to_csv(score_path + "score_{}d{}m{}h.csv".format(now.day, now.month, now.hour), index = None, float_format = "%.9f")
	return print("Score saved in {}".format(score_path))

def main():
	_train, _test, _labels = train, test, labels
	_train, _test = custom_imputation(train, test, fillna_value = 0)
	print("# Start training")
	#split data
	#_train, _validation, _labels, _validation_labels = train_test_split(_train, labels, test_size = 0.1, random_state = 42)
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
	xgb = XGBClassifier(max_depth = 3, n_estimators = 450, subsample = 0.9,
	                    colsample_bytree = 0.8, learning_rate = 0.1)
	#rdforest = RandomForestClassifier(n_jobs = -1)
	#grdboost = GradientBoostingClassifier(n_jobs = -1)

	# ###########################Tuning Params################################
	"""
	params = [
              [{
               #"tbfs__min_child_weight" : [1,2,3], 
               "xgb__max_depth" : [3, 4], 
               "xgb__min_child_weight" : [1, 2, 3],
              }],
           
              [{
               #"tbfs__subsample" : [0.9, 0.8], 
               "xgb__gamma" : [0.1, 0.2], 
               "xgb__subsample" : [0.8, 0.7], 
               "xgb__colsample_bytree" : [0.8, 0.7],
              }],

              [{
              #"tbfs__colsample_bytree" : [0.8, 0.7], 
              "xgb__reg_alpha" : [0.01, 0.03], 
              "xgb__scale_pos_weight" : [1, 10],
              }],

              [{
              "xgb__learning_rate" : [i*0.01 for i in range(3,8)]
              }]

             ]
	"""
	params = [
              [{
               "kbest__k" : [20,40,60], 
               "xgb__max_depth" : [3, 4], 
               "xgb__min_child_weight" : [1, 2, 3],
               #"xgb__max_depth" : [400,450,480]
              }],
           
              [{ 
               "xgb__gamma" : [0.1, 0.2], 
               "xgb__subsample" : [0.8, 0.7], 
               "xgb__colsample_bytree" : [0.8, 0.7],
              }],

              [{
              "xgb__reg_alpha" : [0.01, 0.03], 
              "xgb__scale_pos_weight" : [1, 10],
              }],

              [{
              "xgb__learning_rate" : [i*0.01 for i in range(3,8)]
              }]

             ]

	#pipe = Pipeline(steps = [ #('imputer', Imputer()),
    	                       #('xgb', xgb)])

    #pipe_3 = Pipeline(steps = [('imputer', Imputer()),
                               #('tbfs', SelectFromModel(tbfs)),
    	                       #('xgb', xgb)])

	#pre_pipe = Pipeline([('standar', standar)])

	pipe = Pipeline([('minmax_std', minmax_std), ('kbest', kbest), ('xgb', xgb)])
	#pipe = Pipeline([('kbest', kbest), ('xgb', xgb)])

	clf_initialize = True
	for param in params:
		
		if clf_initialize:
			start = time.time()
			print("\n# Tuning hyper-parameters for {}\n {}".format(param,str("##"*40)))

			clf = GridSearchCV(pipe, param_grid  = param, scoring = 'roc_auc',
			                   verbose = 1, n_jobs = 1, cv = 3)
			#clf = make_pipeline(minmax_std, GridSearchCV(pipe, param_grid  = param, scoring = 'roc_auc',
			                   #verbose = 1, n_jobs = 1, cv = 3))
			clf.fit(_train, _labels)
			bst_params = clf.best_params_
			bst_score = clf.best_score_
			bst_estimator = clf.best_estimator_
			print("\n# Best params set found on development set:\n{}".format(bst_params))
			print(bst_estimator)
			print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
			clf_initialize = False

		else:
			start = time.time()
			print("\n# Tuning hyper-parameters for {}\n {}".format(param,str("##"*40)))
			clf = GridSearchCV(bst_estimator, param_grid  = param, scoring = 'roc_auc',
					verbose = 1, n_jobs = 1, cv = 3)
			#clf = make_pipeline(minmax_std, GridSearchCV(bst_estimator, param_grid  = param, scoring = 'roc_auc',
								#verbose = 1, n_jobs = 1, cv = 3))
			clf.fit(_train, _labels)
			bst_params = clf.best_params_
			bst_score = clf.best_score_
			bst_estimator = clf.best_estimator_
			print("\n# Best params set found on development set:\n{}".format(bst_params))
			print(bst_estimator)
			print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))


	#thresholds = np.sort(clf.best_estimator_.named_steps["xgb"].feature_importances_)
	#thresholds = thresholds.tolist()[int(len(thresholds)*0.4):]
	#print(thresholds)

	"""
	for thresh in thresholds:
		#seletc features using thresh
		selection = SelectFromModel(bst_estimator.steps[0][1], threshold = thresh, prefit = True)
		selected_train = selection.transform(_train)
		selected_val = selection.transform(_validation)
		selected_test = selection.transform(_test)

		#train model
		clf = clf.fit(selected_train, _labels)
		#evl model
		val_preds = clf.predict_proba(selected_val)
		accuracy = accuracy_score(_validation_labels, val_preds)
		print("Tresh = %.3f, Accuracy: %.2f" %(thresh, accuracy * 100))
		#print(matrix)!!!
	"""
		#save score
	probs = clf.predict_proba(_test) #selected_test
	save_score(probs[:,1])

    # ###############################save params################################
	
	with open(params_path  + "params.txt", 'a') as f:
		f.write(
				"************************" + "\n"
				+ str(bst_estimator) + "\n"
				+ str(bst_params) + "\n"
				+ str(bst_score)
				)
	"""
    print("Find best params {}, with best roc {}".format(bst_params, bst_score))
    plot.grid_search(clf.grid_scores_, change='xgb__learning_rate', kind ='bar')
    plt.savefig(params_path + "grid_params_{}.png".format(round(bst_score,2)))
	"""

if __name__ == '__main__':
    warnings.filterwarnings(module = 'sklearn*',
                            action = 'ignore', category = DeprecationWarning)
    os.makedirs(score_path)
    os.makedirs(params_path)
    main()
