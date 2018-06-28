import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import datetime, time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from sklearn_evaluation import plot
import operator
from shutil import rmtree
from hparams import *
from offline_test_search import *
import warnings

def custom_imputation(df_train, df_test, fillna_value = 0):
	train = df_train.fillna(fillna_value)
	test = df_test.fillna(fillna_value)
	print("##"*50)
	print("\n# Filling missing data with <<<{}>>>".format(fillna_value))
	return train, test

def custom_gridsearch(_train, _labels, pipe_clf, param):

	start = time.time()
	print("\n{}\n# Tuning hyper-parameters for {}\n{}\n".format(str("##"*50),param,str("##"*50)))
	clf = GridSearchCV(pipe_clf, param_grid  = param, scoring = 'roc_auc',
	                   verbose = 1, n_jobs = 1, cv = 3)
	clf.fit(_train, _labels)
	bst_params = clf.best_params_
	bst_score = clf.best_score_
	bst_estimator = clf.best_estimator_
	print("\n# Best params set found on development set:\n{}\n{}\n{}".format(str("**"*50),bst_params,str("**"*50)))
	print("\n# Find best estimator \n\n{}\n\n# with best roc {}".format(bst_estimator.steps[-1], bst_score))

	try:
		# ###############################save params################################
		plot_x = next(iter(param[0].keys()))
		plt.figure(figsize=(70,50))
		plot.grid_search(clf.grid_scores_, change= str(plot_x), kind ='bar')
		plt.legend(fontsize=50)
		plt.xlabel(str(plot_x), fontsize = 70)
		plt.ylabel("mean score", fontsize = 70)
		plt.tick_params(axis='both', labelsize = '70')
		plt.savefig(params_path + "{}_{}.png".format(plot_x, round(bst_score,2)))


	except TypeError as e:
		print("\n# Only one param in {}".format(plot_x))
		pass

	with open(params_path  + "params.txt", 'a') as f:
		f.write(
				"**"*40 + "\n"*2
				+ str(bst_estimator.steps[-2:]) + "\n"*2
				+ "Tuned Params : " + str(bst_params) + "\n"*2
				+ "Best ROC : " + str(bst_score) + "\n"*2
				+"**"*40 + "\n"*2
				)

	print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
	print("\n# Cleared est cache ")

	clf_initialize = False

	rmtree(cachedir)
	return clf_initialize, bst_estimator

def save_score(preds):
	answer_sheet = pd.read_csv(as_path)
	answer_sheet = pd.DataFrame(answer_sheet)
	answer = answer_sheet.assign(score = preds)
	answer.to_csv(score_path + "score_day{}_time{}:{}.csv".format(now.day, now.hour, now.minute), index = None, float_format = "%.9f")
	return print("\n# Score saved in {}".format(score_path))

def main(method, _train, _labels, _test_online, _test_offline, _test_offline_labels, fillna_value):

	# #######################Make project path##################################
	warnings.filterwarnings(module = 'sklearn*',
	                        action = 'ignore', category = DeprecationWarning)
	os.makedirs(log_path)
	os.makedirs(score_path)
	os.makedirs(params_path)
	os.makedirs(model_path)
	with open(params_path  + "params.txt", 'a') as f:
		f.write(
				"**"*40 + "\n"*2
				+"Filling missing data with <<<{}>>>".format(str(fillna_value)) + "\n"
				+"Workflow in the order as <<<{}>>>".format(str(strategy[method][2])) + "\n"*2
				+"**"*40 + "\n"*2
				)
	print("\n# Workflow in the order as {}".format(strategy[method][2]))
	Hparams = strategy[method][0]
	pipe = strategy[method][1]

	# #########################GridSearch ######################################
	print("\n# Start training")
	clf_initialize = True
	for param in Hparams:
		if clf_initialize:
			clf_initialize, best_est = custom_gridsearch(_train, _labels, pipe, param)
		else:
			_, best_est = custom_gridsearch(_train, _labels, best_est, param)
	#save model, score
	joblib.dump(best_est, model_path + "{}.pkl".format(method))
	offline_score = offline_model_performance(best_est, _test_offline, _test_offline_labels)
	print("\n# Best perfromance : ", offline_score)
	probs = best_est.predict_proba(_test_online) #selected_test
	save_score(probs[:,1])
