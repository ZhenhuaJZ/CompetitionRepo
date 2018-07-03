import pandas as pd
from sklearn.externals import joblib
from lib.data_processing import *
from lib.model_performance import *
import datetime, time
import sys

def segmentation_model(clf, data, test, feature_dic):
    # Segment data and test into segment a and b
    seg_a_train, seg_b_train = sample_segmentation(data,feature_dic)
    seg_a_test, seg_b_test = sample_segmentation(test, feature_dic)
    # Extract id
    seg_a_score = pd.DataFrame(seg_a_test["id"], columns = ["id"])
    seg_b_score = pd.DataFrame(seg_b_test["id"], columns = ["id"])
    # Segment A data sets
    seg_a_feature, seg_a_label = split_train_label(seg_a_train)
    # Segment B data sets
    seg_b_feature, seg_b_label = split_train_label(seg_b_train)
    ###################### Segment A train and test ####################
    print("\n# Initiate training for segment a")
    seg_a_clf = clf.fit(seg_a_feature, seg_a_label)
    seg_a_test_score = clf.predict_proba(seg_a_test.iloc[:,2:])[:,1]
    print(len(seg_a_score))
    print(seg_a_test_score)
    seg_a_score = seg_a_score.assign(score = seg_a_test_score)
    ###################### Segment B train and test ####################
    print("\n# Initiate training for segment b")
    seg_b_clf = clf.fit(seg_b_feature, seg_b_label)
    seg_b_test_score = clf.predict_proba(seg_b_test.iloc[:,2:])[:,1]
    seg_b_score = seg_b_score.assign(score = seg_b_test_score)

    final_score = seg_a_score.append(seg_b_score)

    return final_score

def positive_unlabel_learning(classifier, unlabel_data, threshold):
	print("\n# PU threshold = {}".format(threshold))
	score = classifier.predict_proba(unlabel_data.iloc[:,2:])
	score = pd.Series(score[:,1])
	score.loc[score >= threshold] = 1
	score.loc[score < threshold] = 0
	unlabel_data.insert(1, "label", score)
	black_unlabel_data = unlabel_data.loc[unlabel_data["label"] == 1]
	n_black = len(unlabel_data[unlabel_data.label == 1])
	n_white = len(unlabel_data[unlabel_data.label == 1])
	print("\n# After PU found <{}> potential black instances, and <{}> potential white instances".format(n_black, n_white))
	clear_mermory(classifier)
	return black_unlabel_data

def partical_fit(data, feed_ratio, sort_by = ""):
	print("\n# Total length :", len(data))
	if sort_by != "":
		data = data.sort_values(by = str(sort_by))
		print("\n# Sort data in <{}> order".format(sort_by))
	partical_loc = int(len(data) * feed_ratio)
	data_seg_1 = data[:partical_loc]
	data_seg_2 = data[partical_loc:]
	print("\n# length of data_seg_1 :", len(data_seg_1))
	print("# length of data_seg_2 :", len(data_seg_2))
	clear_mermory(data)
	return data_seg_1, data_seg_2

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
		cv_offline_score_1 = offline_model_performance(test_offline_labels, offline_probs[:,1], params_path = params_path, fold = i)
		cv_offline_score_2 = offline_model_performance_2(test_offline_labels, offline_probs[:,1], params_path = params_path, fold = i)
		roc_1_list.append(cv_offline_score_1)
		roc_2_list.append(cv_offline_score_2)
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
	#print("\n# ROC_2 :{} (+/- {:2f})".format(roc_2_mean, roc_2_std*2))
	print("##"*40)
	return roc_1_mean, roc_2_mean

def core(fillna, log_path, offline_validation, clf, train_path, test_path, test_a_path, pu_thres, cv = False, fold_time_split = None, under_samp = False, part_fit = True, partical_ratio = 0.5):
	params_path = log_path + "params/"
	score_path = log_path + "score/"
	model_path = log_path + "model/"
	print("\n# Filling missing data with <{}>".format(fillna))
	# ##########################Edit data####################################
	_train_data = pd.read_csv(train_path)
	_train_data = custom_imputation(_train_data)
	_train_data.loc[_train_data["label"] == -1] = 1 #change -1 label to 1
	#Split train and offine test
	_train_data, _test_offline =  test_train_split_by_date(_train_data, offline_validation[0], offline_validation[1], params_path)

	#under_sampling
	if under_samp:
		_train_data = under_sampling(_train_data)

	_train, _labels = split_train_label(_train_data)
	#offline data
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

	# ##########################PU Learning#####################################
	print("\n# *******************PU Traing Start*****************************")
	# NOTE: PU learning
	_test_a = df_read_and_fillna(test_a_path, fillna)
	pu_black_data = positive_unlabel_learning(clf, _test_a, pu_thres)
	clear_mermory(_test_a)
	pu_train_data = file_merge(_train_data, pu_black_data, "date")
	clear_mermory(_train_data, pu_black_data)
	_new_train, _new_label = split_train_label(pu_train_data)
	clf = clf.fit(_new_train, _new_label)
	clear_mermory(_new_train, _new_label)
	print("\n# ********************PU Traing Done*****************************")

	roc_1_mean, roc_2_mean = "n/a","n/a"
	if cv:
		cv_clf = clf
		print("\n# 5-Fold CV (Evaluation Classifier)")
		roc_1_mean, roc_2_mean = cv_fold(cv_clf, pu_train_data, fold_time_split, params_path)

	offline_probs = clf.predict_proba(_test_offline_feature)
	#evl pu model
	offline_score_1 = offline_model_performance(_test_offline_labels, offline_probs[:,1], params_path = params_path)
	offline_score_2 = offline_model_performance_2(_test_offline_labels, offline_probs[:,1], params_path = params_path)
	clear_mermory(_test_offline_feature, _test_offline_labels, offline_probs)

	############################Feed val black back#############################
	# NOTE:  Feed validation black label Back
	print("\n# Feed Only black instances from the validation set to the dataset")
	_test_offline_black = _test_offline.loc[_test_offline["label"] == 1]
	print("\n# Found <{}> black instances".format(len(_test_offline_black)))
	_final_train = file_merge(pu_train_data, _test_offline_black, "date")
	clear_mermory(_test_offline_black, pu_train_data, _test_offline)
	_final_feature, _final_label = split_train_label(_final_train)
	#clear_mermory(_final_train)
	clf = clf.fit(_final_feature, _final_label)
	clear_mermory(_final_feature, _final_label)
	_test_online = df_read_and_fillna(test_path, fillna)

	if not part_fit:
		prob = predict_proba(_test_online.iloc[:,2])
		save_score(prob[:1], score_path)

	if part_fit:
		##########################Partical_fit######################################
		# NOTE:  PU test_b
		#Feed test online
		print("\n# Partical fit <test_b> to the dataset")
		_test_online = df_read_and_fillna(test_path, fillna)
		test_b_seg_1,  test_b_seg_2 = partical_fit(_test_online, partical_ratio, "date")

		#Predict and save seg_1 score
		prob_seg_1 = clf.predict_proba(test_b_seg_1.iloc[:,2:])
		score_seg_1 = pd.DataFrame(test_b_seg_1["id"]).assign(score = prob_seg_1[:,1])

		clear_mermory(_test_online)
		test_b_seg_1_black = positive_unlabel_learning(clf, test_b_seg_1, 0.5) #pu threshold
		clear_mermory(test_b_seg_1)
		increment_train = file_merge(test_b_seg_1_black, _final_train, "date")
		increment_train.to_csv("data/increment_train.csv", index = None)
		print("\n# Partical fit done !")
		clear_mermory(test_b_seg_1_black, _final_train)
		"""
		#########################Merge Test_b score#################################
		increment_train_feature, increment_train_label = split_train_label(increment_train)
		clear_mermory(increment_train)
		#Fit new classifier
		clf.fit(increment_train_feature, increment_train_label)

		#Predict and save seg_2 score
		prob_seg_2 = clf.predict_proba(test_b_seg_2.iloc[:,2:])
		score_seg_2 = pd.DataFrame(test_b_seg_2["id"]).assign(score = prob_seg_2[:,1])

		##############################Merge Score###################################
		score = score_seg_1.append(score_seg_2)
		score.to_csv(score_path + "score_day{}_time{}:{}.csv".format(now.day, now.hour, now.minute), index = None, float_format = "%.9f")
		print("\n# Score saved in {}".format(score_path))
		"""
	#Log all the data
	log_parmas(clf, valset = offline_validation,
				roc_1 = offline_score_1, roc_2 = offline_score_2,
				CV_ROC_1 = roc_1_mean, CV_ROC_2 = roc_2_mean, Score = "",
				score_path = score_path, pu_thres = pu_thres, partical_fit = partical_fit,
				under_samp = under_samp, fillna = fillna)

	clear_mermory(now)
