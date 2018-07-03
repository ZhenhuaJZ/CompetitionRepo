import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from lib.data_processing import *
from lib.model_performance import *
from lib.tool import *
from core_model import *
import datetime, time

# #####################Data path###########################################
train_path = "data/train.csv" #train_heatmap , train,
test_path = "data/test_b.csv" #test_a_heatmap, test_b
test_a_path = "data/test_a.csv"

def main():
	# #####################################################################

	fillna = 0
	clf_name = "XGB" #LR,MLP,RF,XGB

	#Tunning params
	tuning_name = "thresh"
	tunning = False
	tuning_range = [0.4,0.6,0.5,0.7,0.3]

	pu_thres = 0.4
	offline_validation = [20171025, 20171105] #20171025, 20171105
	#CV
	cv = True
	fold_time_split = [[20170905, 20170916], [20170917, 20170925], [20170926, 20171005],[20171006,20171015],[20171015,20171025]]
	#under_sampling
	under_samp = False
	method = "pu_method"
	"""
	command = {
				"fillna" : 0,
				"clf_name" : "XGB", #LR,MLP,RF,XGB
				"tuning_name" : "pu_thresh",
				#Tunning params
				#"tunning" : True,
				"tuning_range" : [0.5, 0.6, 0.4, 0.3, 0.2, 0.1],
				#Method
				#"method" : "pu_method", #pu_method, single_mode
				"pu_thres" : 0.5,
				"offline_validation" : [20171025, 20171105], #20171025, 20171105
				#CV
				#"cv" : True,
				"fold_time_split" : [[20170905, 20170910], [20170911, 20170920], [20170921, 20171001],[20171002,20171015],[20171015,20171027]],
				#under_sampling
				"under_samp" : False,
			  }

	#double_check(command)

	#clf_name = command["clf_name"]
	#tuning_name = command["tuning_name"]
	#tuning_range = command["tuning_range"]
	#pu_thres = command["pu_thres"]
	#offline_validation = command["offline_validation"]
	#fold_time_split = command["fold_time_split"]
	#under_samp = command["under_samp"]
	"""
	if tunning:
		for p in tuning_range:
			classifier = {
			"XGB" : XGBClassifier(max_depth = 4, n_estimators = 4, subsample = 0.8, gamma = 0,
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
			print("\n"+ "##"*40 + "\n# Tuning : {} current at {}".format(tuning_name, p))
			clf = classifier[clf_name]
			now = datetime.datetime.now()
			log_path = "log/date_{}/Tuning_{}_{}/{}:{}_GS/".format(now.day, clf_name, tuning_name, now.hour,now.minute)
			creat_project_dirs(log_path)
			core(fillna, log_path, offline_validation, clf, train_path, test_path, test_a_path,
					pu_thres = p, method = method, cv = cv, fold_time_split = fold_time_split, under_samp = under_samp)
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
		core(fillna, log_path, offline_validation, clf, train_path, test_path, test_a_path,
		 			pu_thres = pu_thres, method = method, cv = cv, fold_time_split = fold_time_split, under_samp = under_samp)

if __name__ == '__main__':
	main()
