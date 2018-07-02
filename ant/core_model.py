import pandas as pd
from sklearn.externals import joblib
from lib.data_processing import *
from lib.model_performance import *
import datetime, time

def feed_validation(classifier, data):
	label_1_data = data.loc[data["label"] == 1]
	_train = file_merge(label_1_data, train)
	return _train

def positive_unlabel_learning(classifier, unlabel_data, threshold):
	print("\n# PU threshold = {}".format(threshold))
	score = classifier.predict_proba(unlabel_data.iloc[:,2:])
	score = pd.Series(score[:,1])
	score.loc[score >= threshold] = 1
	score.loc[score < threshold] = 0
	unlabel_data.insert(1, "label", score)
	print("\n# After PU found <{}> potential black instances".format(len(unlabel_data[unlabel_data.label == 1])))
	print("\n# After PU found <{}> potential white instances".format(len(unlabel_data[unlabel_data.label == 0])))
	return unlabel_data

def cv_fold(clf, _train_data, fold_time_split, params_path):
	roc_1_list = []
	roc_2_list = []
	for i, offline_validation in enumerate(fold_time_split):
		#CV in 5 fold
		start = time.time()
		print("\n"+"##"*40)
		print("\n# Fold {} from {} - {}".format(i, offline_validation[0], offline_validation[1]))
		print("\n"+"##"*40)
		train_data, test_offline =  test_train_split_by_date(_train_data, offline_validation[0], offline_validation[1], params_path)
		train, labels = split_train_label(train_data)
		test_offline_feature, test_offline_labels = split_train_label(test_offline)
		clear_mermory(test_offline, train_data)
		#Fead data into the clf
		_clf = clf.fit(train, labels)
		clear_mermory(train, labels)
		offline_probs = _clf.predict_proba(test_offline_feature)
		clear_mermory(test_offline, _clf)
		offline_score_1 = offline_model_performance(test_offline_labels, offline_probs[:,1], params_path = params_path, fold = i)
		offline_score_2 = offline_model_performance_2(test_offline_labels, offline_probs[:,1], params_path = params_path, fold = i)
		roc_1_list.append(offline_score_1)
		roc_2_list.append(offline_score_2)
		print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
	#eval performace
	#clear_mermory(_train_data)
	roc_1 = np.array(roc_1_list)
	roc_2 = np.array(roc_2_list)
	roc_1_mean = np.mean(roc_1, axis = 0)
	roc_2_mean = np.mean(roc_2, axis = 0)
	roc_1_std = np.std(roc_2, axis = 0)
	roc_2_std = np.std(roc_1, axis = 0)
	print("##"*40)
	print("\n# ROC_1(JL) :{} (+/- {:2f})".format(roc_1_mean, roc_1_std*2))
	print("\n# ROC_2 :{} (+/- {:2f})".format(roc_2_mean, roc_2_std*2))
	print("##"*40)
	return roc_1_mean, roc_2_mean

def core(fillna, log_path, offline_validation, clf, train_path, test_path, test_a_path, pu_thres, method = None, cv = False, fold_time_split = None, under_samp = False):
	params_path = log_path + "params/"
	score_path = log_path + "score/"
	model_path = log_path + "model/"
	# ##########################Edit data####################################
	_train_data = pd.read_csv(train_path)
	_test_online = pd.read_csv(test_path)
	_test_a = pd.read_csv(test_a_path)

	_train_data, _test_online, _test_a = custom_imputation_3_inputs(_train_data, _test_online, _test_a, fillna)
	#change -1 label to 1
	_train_data.loc[_train_data["label"] == -1] = 1

	#Split train and offine test
	_train_data, _test_offline =  test_train_split_by_date(_train_data, offline_validation[0], offline_validation[1], params_path)

	#under_sampling
	if under_samp:
		_train_data = under_sampling(_train_data)

	_train, _labels = split_train_label(_train_data)
	#online & offline data
	_test_online = _test_online.iloc[:,2:]
	_test_offline_feature, _test_offline_labels = split_train_label(_test_offline)

	# ##########################Traing model####################################
	start = time.time()
	with open(params_path  + "params.txt", 'a') as f:
		print("\n# Training clf :{}".format(clf))
		f.write(
		"**"*40 + "\n"*2
		+ str(clf) + "\n"*2
		+"**"*40 + "\n"*2
		)

	clf = clf.fit(_train, _labels)
	clear_mermory(_train, _labels)
	print(clf.get_params())
	print("\n# PU Traing Start")
	# NOTE: PU learning
	unlabel_data = positive_unlabel_learning(clf, _test_a, pu_thres)
	clear_mermory(_test_a)
	#Choose Black Label
	unlabel_data = unlabel_data[unlabel_data.label == 1]
	pu_train_data = file_merge(_train_data, unlabel_data, "date")
	clear_mermory(_train_data, unlabel_data)
	_new_train, _new_label = split_train_label(pu_train_data)
	clf = clf.fit(_new_train, _new_label)
	clear_mermory(_new_train, _new_label)
	print("\n# PU Traing Done")

	roc_1_mean, roc_2_mean = "n/a","n/a"
	if cv:
		if method == "pu_method" :
			print("\n# 5 - Fold CV for PU classifier (Evaluation Classifier)")
		roc_1_mean, roc_2_mean = cv_fold(clf, _train_data, fold_time_split, params_path)

	offline_probs = clf.predict_proba(_test_offline_feature)
	print(clf.get_params())
	#evl pu model
	offline_score_1 = offline_model_performance(_test_offline_labels, offline_probs[:,1], params_path = params_path)
	offline_score_2 = offline_model_performance_2(_test_offline_labels, offline_probs[:,1], params_path = params_path)
	clear_mermory(_test_offline_feature, _test_offline_labels, offline_probs)
	#probs = clf.predict_proba(_test_online)
	#save_score(probs[:,1], score_path)
	#log_parmas(clf, offline_validation, offline_score_1, offline_score_2, method, log_path, fillna, pu_thres, roc_1_mean, roc_1_mean, under_samp)
	#clear_mermory(now)

	# NOTE:  Feed validation black label Back
	print("\n# Feed Only black validation set to the dataset")
	_test_offline_black = _test_offline.loc[_test_offline["label"] == 1]
	print("\n# Found <{}> black instances".format(len(_test_offline_black)))
	_final_train = file_merge(pu_train_data, _test_offline_black, "date")
	clear_mermory(_test_offline_black, pu_train_data, _test_offline)
	_final_feature, _final_label = split_train_label(_final_train)
	clear_mermory(_final_train)
	#joblib.dump(clf, model_path + "{}.pkl".format("model"))
	clf = clf.fit(_final_feature, _final_label)
	print(clf.get_params())
	clear_mermory(_final_train, _final_label)
	probs = clf.predict_proba(_test_online)
	save_score(probs[:,1], score_path)

	#Log all the data
	log_parmas(clf, offline_validation, offline_score_1, offline_score_2, method,
				log_path, fillna, pu_thres, roc_1_mean, roc_1_mean, under_samp)
	clear_mermory(now)
	print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
