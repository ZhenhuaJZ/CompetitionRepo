import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from lib.data_processing import *
from lib.model_performance import *
from sklearn.base import clone
import datetime, time
import sys
from copy import copy
now = datetime.datetime.now()

def grid_search_roc(clf, train, test, params):
	feature, label = split_train_label(train)
	test_feature, test_label = split_train_label(test)
	best_clf = clone(clf)
	best_auc = 0
	best_param = {}
	start = time.time()
	for para in params:
		for i in params[para]:
			clf = clone(best_clf)
			parameter = {para : i}
			print("\n#"+"*"*20+" Current parameter: {}".format(parameter)+" "+"*"*20)
			print("\n# Current best parameter ", best_param)
			clf.set_params(**parameter)
			clf.fit(feature,label)
			score = clf.predict_proba(test_feature)[:,1]
			auc = offline_model_performance_2(test_label,score)
			print("\n# AUC offline performance : {}".format(auc))
			if auc > best_auc:
				print("# Best paramter found")
				best_clf = clone(clf)
				best_auc = auc
				best_param["{}".format(para)] = i
			print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

	print(best_clf)
	return best_clf

def test_b_grid_search(clf, train_pu_a, validation, threshold_range, seg_date, params, log_path = ""):
	test_b = pd.read_csv(test_b_path)
	test_b = test_b.loc[:100000]
	# print(test_b)
	_feature, _label = split_train_label(train_pu_a)
	validation_feature, validation_label = split_train_label(validation)
	test_b_seg_1, _ = partical_fit(test_b, seg_date, "date")
	best_clf = clone(clf)
	best_auc = 0
	best_param = {}
	for threshold in threshold_range:
		for para in params:
			for i in params[para]:
				clf = clone(best_clf)
				parameter = {para : i}
				test_b_1 = copy(test_b_seg_1)
				print("\n#"+"*"*20+" Current parameter: {}".format(parameter)+" "+"*"*20)
				print("\n# Current threshold: {}".format(threshold))
				print("\n# Current best parameter ", best_param)
				clf.set_params(**parameter)
				clf.fit(_feature, _label)
				probs = clf.predict_proba(test_b_1.iloc[:,2:])
				test_b_seg_1_black = positive_unlabel_learning(clf, test_b_1, threshold) #pu threshold
				pu_b_train = file_merge(train_pu_a, test_b_seg_1_black, "date")
				pu_b_feature, pu_b_label = split_train_label(pu_b_train)
				clf.fit(pu_b_feature, pu_b_label)
				validation_score = clf.predict_proba(validation_feature)[:,1]
				auc = offline_model_performance_2(validation_label, validation_score)
				print("\n# AUC offline performance : {}".format(auc))
				if auc > best_auc:
					print("# Best paramter found")
					best_param["threshold"] = threshold
					best_clf = clone(clf)
					best_auc = auc
					best_thresh = threshold
					best_param["{}".format(para)] = i
	if log_path != "":
		log_parmas(best_clf, log_path, best_thresh = best_thresh, best_auc = best_auc)
	print(best_clf)
	print(best_auc)
	print(best_thresh)

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

def pu_labeling(classifier, unlabel, threshold):

	score = classifier.predict_proba(unlabel.iloc[:,2:])
	score = pd.Series(score[:,1], index = unlabel["f1"].index)

	score.loc[score >= threshold] = 1
	score.loc[score < threshold] = 0
	unlabel.insert(1, "label", score)

	black = unlabel.loc[unlabel["label"] == 1]
	n_black = len(unlabel[unlabel.label == 1])
	n_white = len(unlabel[unlabel.label == 0])
	print("\n# After PU found <{}> potential black instances, and <{}> potential white instances".format(n_black, n_white))
	clear_mermory(classifier)
	return black

def partical_fit(data, start_y_m_d, sort_by = ""):
    print("\n# Total length :", len(data))
    if sort_by != "":
        data = data.sort_values(by = str(sort_by))
        print("\n# Sort data in <{}> order".format(sort_by))
    data_seg_1 = data[(data["date"] < start_y_m_d)]
    data_seg_2 = data[(data["date"] >= start_y_m_d)]
    #partical_loc = int(len(data) * feed_ratio)
    #data_seg_1 = data[:partical_loc]
    #data_seg_2 = data[partical_loc:]
    print("\n# length of data_seg_1 :", len(data_seg_1))
    print("# length of data_seg_2 :", len(data_seg_2))
    clear_mermory(data)
    return data_seg_1, data_seg_2

def cv_fold(clf, train, slice_interval):
	roc_list = []
	for i, fold in enumerate(slice_interval):
		#CV in 5 fold
		start = time.time()
		print("\n"+"##"*40)
		print("\n# Fold {} from {} - {}".format(i, fold[0], fold[1]))
		print("\n"+"##"*40)
		train_data, test_offline =  test_train_split_by_date(train, fold[0], fold[1])
		_feature_train, _label_train = split_train_label(train_data)
		_feature_val, _label_val = split_train_label(test_offline)
		_clf = clf.fit(_feature_train, _label_train)
		probs = _clf.predict_proba(_feature_val)
		roc = offline_model_performance(_label_val, probs[:,1])
		roc_list.append(roc)
		print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
	#eval performace
	roc = np.array(roc_list)
	roc_mean = np.mean(roc, axis = 0)
	roc_std = np.std(roc, axis = 0)
	print("##"*40)
	print("\n# ROC_1(JL) :{} (+/- {:2f})".format(roc_mean, roc_std*2))
	print("##"*40)
	return roc_mean

def core(fillna, log_path, offline_validation, clf, train_path, test_path, test_a_path, pu_thres, cv = False, fold_time_split = None, under_samp = False, part_fit = True, partical_ratio = 0.5):

    params_path = log_path + "params/"
    score_path = log_path + "score/"
    model_path = log_path + "model/"

    # ##########################Edit data#######################################
    _train_data = pd.read_csv(train_path)
    #change -1 label to 1
    _train_data.loc[_train_data["label"] == -1] = 1 #change -1 label to 1
    #_train_data = _train_data.replace({"label" : -1}, value = 1)
    #Split train and offine test
    _train_data, _test_offline =  test_train_split_by_date(_train_data, offline_validation[0], offline_validation[1])

    _train_data.to_csv("_train_data_adding_1.csv", index = None)
    _test_offline.to_csv("_test_offline_adding_1.csv", index = None)
    sys.exit()
    #under_sampling
    if under_samp:
        print("\n# Under_sampling")
        _train_data = under_sampling(_train_data)

    _train, _labels = split_train_label(_train_data)
    _test_offline_feature, _test_offline_labels = split_train_label(_test_offline)

    # ##########################Traing model####################################
    start = time.time()
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
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    roc_1_mean, roc_2_mean = "n/a","n/a"
    if cv:
        cv_clf = clf
        print("\n# 5-Fold CV (Evaluation Classifier)")
        roc_1_mean, roc_2_mean = cv_fold(cv_clf, pu_train_data, fold_time_split)

    offline_probs = clf.predict_proba(_test_offline_feature)
    #evl pu model
    offline_score_1 = offline_model_performance(_test_offline_labels, offline_probs[:,1])
    #offline_score_2 = offline_model_performance_2(_test_offline_labels, offline_probs[:,1])
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
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
    #_test_online = df_read_and_fillna(test_path, fillna)

    if not part_fit:
        prob = clf.predict_proba(_test_online.iloc[:,2:])
        save_score(prob[:,1], score_path)

    if part_fit:
        ##########################Partical_fit######################################
        # NOTE:  PU test_b
        #Feed test online
        print("\n# Partical fit {} * <test_b> to the dataset".format(partical_ratio))
        _test_online = df_read_and_fillna(test_path, fillna)

        test_b_seg_1,  test_b_seg_2 = partical_fit(_test_online, partical_ratio, "date")

        #Predict and save seg_1 score
        prob_seg_1 = clf.predict_proba(test_b_seg_1.iloc[:,2:])
        score_seg_1 = pd.DataFrame(test_b_seg_1["id"]).assign(score = prob_seg_1[:,1])
        #score_seg_1_path = score_path + "score_seg_a.csv"
        #score_seg_1.to_csv(score_seg_1_path) # delete index for testing
        #print("\n# Parrical_score_1 saved in path {} !".format(score_seg_1_path))
        clear_mermory(_test_online)

        #PU for test_b
        test_b_seg_1_black = positive_unlabel_learning(clf, test_b_seg_1, 0.97) #pu threshold
        clear_mermory(test_b_seg_1)
        increment_train = file_merge(test_b_seg_1_black, _final_train, "date")
        clear_mermory(test_b_seg_1_black, _final_train)
        #increment_train_path = log_path + "increment_train.csv"

        increment_train_feature, increment_train_label = split_train_label(increment_train)
        clear_mermory(increment_train)
        clf = clf.fit(increment_train_feature, increment_train_label)
        print("\n# Partical fit done !")
        print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

        clear_mermory(increment_train_feature, increment_train_label)
        prob_seg_2 = clf.predict_proba(test_b_seg_2.iloc[:,2:])
        score_seg_2 = pd.DataFrame(test_b_seg_2["id"]).assign(score = prob_seg_2[:,1])
        score = score_seg_1.append(score_seg_2).sort_index()
        score.to_csv(score_path + "score_day{}_time{}:{}.csv".format(now.day, now.hour, now.minute), index = None, float_format = "%.9f") #delete index for testing
        print("\n# Score saved in {}".format(score_path))
        print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
        #Save increment_train to hard drive
        #increment_train.to_csv(increment_train_path, index = None)

        #print("# incremental train data saved in path {} !".format(increment_train_path))

        """
        #Read increment_train from hard drive
        print("\n# Inititalize increment_train")
        #_train_data = pd.read_csv(increment_train_path, low_memory = False)
        #increment_train = df_read_and_fillna(increment_train_path)

        #########################Merge Test_b score#################################
        increment_train_feature, increment_train_label = split_train_label(increment_train)
        clear_mermory(increment_train)
        #Fit new classifier
        clf.fit(increment_train_feature, increment_train_label)
        clear_mermory(increment_train_feature, increment_train_label)

        #Predict and save seg_2 score
        prob_seg_2 = clf.predict_proba(test_b_seg_2.iloc[:,2:])
        score_seg_2 = pd.DataFrame(test_b_seg_2["id"]).assign(score = prob_seg_2[:,1])

        ##############################Merge Score###################################
        score_seg_1 = pd.read_csv(score_seg_1_path)
        score = score_seg_1.append(score_seg_2)
        score.to_csv(score_path + "score_day{}_time{}:{}.csv".format(now.day, now.hour, now.minute), float_format = "%.9f") #delete index for testing
        print("\n# Score saved in {}".format(score_path))
        """
	#Log all the data
    """
    log_parmas(clf, valset = offline_validation,
        roc_1 = offline_score_1, roc_2 = offline_score_2,
        CV_ROC_1 = roc_1_mean, CV_ROC_2 = roc_2_mean, Score = "",
        score_path = score_path, pu_thres = pu_thres, partical_fit = partical_fit,
        under_samp = under_samp, fillna = fillna)
    """
    clear_mermory(now)

def eval_test_set(clf, test_set):
	_feature, _label = split_train_label(test_set)
	print(_feature)
	print(_label)
	probs = clf.predict_proba(_feature)
	roc = offline_model_performance(_label, probs[:,1])
	return roc

def eval_validation_set(clf, train_set):
    _day = []
    interval = int(len(train_set["date"])/5)
    for i in range(6):
        _day.append(train_set["date"].iloc[interval*i])
    slice_interval = [[_day[0], _day[1]], [_day[1]+1, _day[2]], [_day[2]+1, _day[3]],[_day[3]+1,_day[4]],[_day[4]+1, _day[5]]]
    roc = cv_fold(clf, train_set, slice_interval)
    return roc

def evaluation(clf, test_set_path, train_set):
    roc_val = eval_validation_set(clf, train_set)
    roc_test =  eval_test_set(clf, test_set_path)
    # TODO:  high variance, low varience .etc
    return roc_val, roc_test
