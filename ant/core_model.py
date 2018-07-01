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


def core(fillna, log_path, offline_validation, method, clf, train_path, test_path, test_a_path):

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

	if method == "single_model":
		clf = clf.fit(_train, _labels)
		clear_mermory(_train, _labels)
		offline_probs = clf.predict_proba(_test_offline_feature)
		clear_mermory(_test_offline)
		offline_score_1 = offline_model_performance(_test_offline_labels, offline_probs[:,1], params_path = params_path)
		offline_score_2 = offline_model_performance_2(_test_offline_labels, offline_probs[:,1], params_path = params_path)
		if offline_score_2 > offline_score_1:
			print("Goog performance_2")
		else:
			print("Goog performance_1(JL)")
		clear_mermory(_test_offline_feature, _test_offline_labels, offline_probs)
		probs = clf.predict_proba(_test_online)
		save_score(probs[:,1], score_path)
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
	elif method == "pu_method" :
		# NOTE: PU learning
		_clf = clf.fit(_train, _labels)
		clear_mermory(_train, _labels)
		#without PU offline score
		offline_probs = clf.predict_proba(_test_offline_feature)
		clear_mermory(_test_offline)
		offline_score = offline_model_performance(_test_offline_labels, offline_probs[:,1], params_path = params_path)
		unlabel_data = positive_unlabel_learning(_clf, _test_a, 0.6)
		#Choose Black Label
		unlabel_data = unlabel_data[unlabel_data.label == 1]
		#80% train data
		pu_train_data = file_merge(_train_data, unlabel_data, "date")
		_new_train, _new_label = split_train_label(pu_train_data)
		#recall clf
		new_clf = clf.fit(_new_train, _new_label)
		clear_mermory(_new_train, _new_label)
		probs = new_clf.predict_proba(_test_online)
		#joblib.dump(clf, model_path + "{}.pkl".format("model"))
		#with PU offline score
		offline_score = offline_model_performance(_test_offline_labels, probs[:,1], params_path = params_path)
		save_score(probs[:,1], score_path)

	log_parmas(clf, offline_validation, offline_score_1, offline_score_2, method, log_path, fillna)
	clear_mermory(now)
	print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
