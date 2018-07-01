import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from lib.data_processing import *
from lib.model_performance import *
from core_model import *
import datetime, time

# #####################Data path###########################################
train_path = "data/train.csv" #train_heatmap , train,
test_path = "data/test_b.csv" #test_a_heatmap, test_b
test_a_path = "data/test_a.csv"

def main():
	# #####################################################################
	#Tunning params
	fillna = 0
	tunning = True
	clf_name = "XGB" #LR,MLP,RF,XGB
	tuning_name = "n_estimators"
	#loop_start, loop_end, loop_step
	#range = np.arange(1, 30, 4)
	range = [400,470,480,485,490]
	method = "single_model" #pu_method
	offline_validation = [20171025, 20171105]

	if tunning:
		for p in range:
			classifier = {
			"XGB" : XGBClassifier(max_depth = 5, n_estimators = p, subsample = 0.8, gamma = 0,
			min_child_weight = 1, scale_pos_weight = 1, reg_alpha = 0,
			colsample_bytree = 0.8, learning_rate = 0.07, n_jobs = -1),

			"LR" : LogisticRegression(#penalty = "l2", C = p, solver = "sag",
			class_weight = "balanced", max_iter = 100, n_jobs = -1),

			# NOTE:test min_samples_split and min_samples_leaf
			"RF" : RandomForestClassifier(n_estimators = 300, criterion = "entropy", max_depth = 16,
			min_samples_split = p, min_samples_leaf = 16, max_leaf_nodes = None,
			n_jobs = -1),

			"MLP" : MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
			beta_1=0.9, beta_2=0.999, early_stopping=False,
			epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
			learning_rate_init=0.001, max_iter=200, momentum=0.9,
			nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
			solver='lbfgs', tol=0.0001, validation_fraction=0.1)
			}
			print("\n# Tuning : {} current at {}".format(tuning_name, p))
			clf = classifier[clf_name]
			now = datetime.datetime.now()
			log_path = "log/date_{}/Tuning_{}_{}/{}:{}_GS/".format(now.day, clf_name, tuning_name, now.hour,now.minute)
			creat_project_dirs(log_path)
			core(fillna, log_path, offline_validation, method, clf, train_path, test_path, test_a_path)
	else:
		classifier = {
		"XGB" : XGBClassifier(max_depth = 4, n_estimators = 4, subsample = 0.8, gamma = 0,
		min_child_weight = 1, scale_pos_weight = 1,
		colsample_bytree = 0.8, learning_rate = 0.07, n_jobs = -1),

		"logistic_regression" : LogisticRegression(penalty = "l2", C = 1, solver = "sag",
		class_weight = "balanced", max_iter = 100, n_jobs = -1),

		# NOTE:test min_samples_split and min_samples_leaf
		"random_forest" : RandomForestClassifier(n_estimators = 300, criterion = "entropy", max_depth = 16,
		min_samples_split = 110, min_samples_leaf = 1, max_leaf_nodes = None,
		n_jobs = -1),

		"MLP" : MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
		beta_1=0.9, beta_2=0.999, early_stopping=False,
		epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
		learning_rate_init=0.001, max_iter=200, momentum=0.9,
		nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
		solver='lbfgs', tol=0.0001, validation_fraction=0.1)
		}

		clf = classifier[clf_name]
		now = datetime.datetime.now()
		log_path = "log/date_{}/{}:{}_SM/".format(now.day,now.hour,now.minute)
		creat_project_dirs(log_path)
		core(fillna, log_path, offline_validation, method, clf, train_path, test_path, test_a_path)

if __name__ == '__main__':
	main()
