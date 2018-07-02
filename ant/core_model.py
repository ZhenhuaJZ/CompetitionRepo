import pandas as pd
from sklearn.externals import joblib
from lib.data_processing import *
from lib.model_performance import *
import datetime, time

def positive_unlabel_learning(classifier, unlabel_data, threshold):
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
		print("\n# Fold {} from {} - {}".format(i, offline_validation[0], offline_validation[1]))
		_train_data, _test_offline =  test_train_split_by_date(_train_data, offline_validation[0], offline_validation[1], params_path)
		_train, _labels = split_train_label(_train_data)
		_test_offline_feature, _test_offline_labels = split_train_label(_test_offline)
		clear_mermory(_train_data, _test_offline)
		#Fead data into the clf
		clf = clf.fit(_train, _labels)
		clear_mermory(_train, _labels)
		offline_probs = clf.predict_proba(_test_offline_feature)
		clear_mermory(_test_offline)
		offline_score_1 = offline_model_performance(_test_offline_labels, offline_probs[:,1], params_path = params_path, fold = i)
		offline_score_2 = offline_model_performance_2(_test_offline_labels, offline_probs[:,1], params_path = params_path, fold = i)
		roc_1 = np.array(roc_1_list.append(offline_score_1))
		roc_2 = np.array(roc_2_list.append(offline_score_2))
	#eval performace
	roc_1_mean = np.mean(roc_1, axis = 0)
	roc_2_mean = np.mean(roc_2, axis = 0)
	roc_1_std = np.std(roc_2, axis = 0)
	roc_2_std = np.std(roc_1, axis = 0)
	print("\n# ROC :{} (+/- {:2f})".format(roc_1_mean, roc_1_std*2))
	print("\n# ROC :{} (+/- {:2f})".format(roc_2_mean, roc_2_std*2))

	return roc_1_mean, roc_2_mean

def core(fillna, log_path, offline_validation, method, clf, train_path, test_path, test_a_path, cv = False):
	fold_time_split = [[20170905, 20170915], [20170916, 20170930], [20171001, 20171015],[20171016,20171031],[20171101,20171105]]
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

	if method == "pu_method" :
		# NOTE: PU learning
		#without PU offline score
		offline_probs = clf.predict_proba(_test_offline_feature)
		clear_mermory(_test_offline)
		offline_model_performance(_test_offline_labels, offline_probs[:,1], params_path = params_path)
		offline_model_performance_2(_test_offline_labels, offline_probs[:,1], params_path = params_path)
		unlabel_data = positive_unlabel_learning(clf, _test_a, 0.6)
		clear_mermory(_test_a)
		#Choose Black Label
		unlabel_data = unlabel_data[unlabel_data.label == 1]
		print(unlabel_data)
		#80% train data
		pu_train_data = file_merge(_train_data, unlabel_data, "date")
		clear_mermory(_train_data, unlabel_data)
		_new_train, _new_label = split_train_label(pu_train_data)
		#recall clf
		clf = clf.fit(_new_train, _new_label)
		clear_mermory(_new_train, _new_label)

	if cv:
		roc_1_mean, roc_2_mean = cv_fold(clf, _train_data, fold_time_split, params_path)

	offline_probs = clf.predict_proba(_test_offline_feature)
	clear_mermory(_test_offline)
	offline_score_1 = offline_model_performance(_test_offline_labels, offline_probs[:,1], params_path = params_path)
	offline_score_2 = offline_model_performance_2(_test_offline_labels, offline_probs[:,1], params_path = params_path)
	clear_mermory(_test_offline_feature, _test_offline_labels, offline_probs)
	probs = clf.predict_proba(_test_online)
	save_score(probs[:,1], score_path)
	#Log all the data
	log_parmas(clf, offline_validation, offline_score_1, offline_score_2, method, log_path, fillna, roc_1_mean, roc_2_mean)
	clear_mermory(now)

	# NOTE:  Feed validation Back
	"""
	print("\n# Feed validation set to the dataset")
	all_train = file_merge(_train_data, _test_offline, "date")
	clear_mermory(_test_offline, _train_data)
	_new_train, _new_label = split_train_label(all_train)
	clear_mermory(all_train)
	#joblib.dump(clf, model_path + "{}.pkl".format("model"))
	new_clf = clf.fit(_new_train, _new_label)
	clear_mermory(_new_train, _new_label)
	probs = new_clf.predict_proba(_test_online)
	save_score(probs[:,1], score_path)
	"""

	print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
