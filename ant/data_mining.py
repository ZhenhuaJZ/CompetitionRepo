import pandas as pd
from xgboost import XGBClassifier
from lib.data_processing import *
from lib.model_performance import *
from core_model import positive_unlabel_learning, partical_fit
import time, sys, datetime
now = datetime.datetime.now()

train_path = "data/_train_data.csv"
test_b_path = "data/test_b.csv"
test_a_path = "data/test_a.csv"
validation_path = "data/_test_offline.csv"
log_path = "log/"

pu_thresh_a = 0.8 #PU threshold for testa
pu_thresh_b = 0.95 #PU threshold for testb
partial_rate = 0.4

def init_train(train_path, eval = True):

    start = time.time()
    clf = XGBClassifier(max_depth = 4, n_estimators = 450, subsample = 0.8, gamma = 0.1,
                    min_child_weight = 1, scale_pos_weight = 1,
                    colsample_bytree = 0.7, learning_rate = 0.07, n_jobs = -1)
    #Train
    print("\n# Start Traing")
    train = pd.read_csv(train_path)
    feature, label = split_train_label(train)
    clf.fit(feature, label)
    clear_mermory(feature, label, train_path)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if eval:

        print("\n# EVAL INIT CLASSIFIER")
        validation_path = "data/_test_offline.csv"
        validation = pd.read_csv(validation_path)
        val_feature, val_label = split_train_label(validation)
        val_probs = clf.predict_proba(val_feature)
        roc = offline_model_performance(val_label, val_probs[:,1])
        clear_mermory(val_feature, val_label, validation, validation_path, val_probs)
        return clf, train, roc

    return clf, train

def positive_unlabel(clf, test_a_path, train, pu_thresh_a, eval = True):
    #PU
    start = time.time()
    print("\n# START PU - TESTA , PU_thresh_a = {}".format(pu_thresh_a))
    test_a = pd.read_csv(test_a_path)
    pu_black = positive_unlabel_learning(clf, test_a, pu_thresh_a)
    pu_train = file_merge(train, pu_black, "date")
    pu_feature, pu_label = split_train_label(pu_train)
    clf.fit(pu_feature, pu_label)
    clear_mermory(pu_feature, pu_label, train, pu_black, test_a_path, test_a)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if eval:

        print("\n# EVAL PU")
        validation_path = "data/_test_offline.csv"
        validation = pd.read_csv(validation_path)
        val_feature, val_label = split_train_label(validation)
        val_probs = clf.predict_proba(val_feature)
        roc = offline_model_performance(val_label, val_probs[:,1])
        clear_mermory(val_feature, val_label, validation, validation_path, val_probs)
        return clf, train, roc

    return clf, pu_train

def part_fit(clf, test_b_path, pu_train, partial_rate, pu_thresh_b):
    #Partical_Fit
    start = time.time()
    print("\n# PART FIT TESTB, PU_thresh_b = {}, Partial_Rate = {}".format(pu_thresh_b, partial_rate))
    test_b = pd.read_csv(test_b_path)
    test_b_seg_1, test_b_seg_2 = partical_fit(test_b, partial_rate, "date")
    probs = clf.predict_proba(test_b_seg_1.iloc[:,2:])
    score_seg_1 = pd.DataFrame(test_b_seg_1["id"]).assign(score = probs[:,1])
    pu_test_b_seg_1 = positive_unlabel_learning(clf, test_b_seg_1, pu_thresh_b) #pu threshold
    incre = file_merge(pu_test_b_seg_1, pu_train, "date")
    incre_feature, incre_label = split_train_label(incre)
    _clf = clf.fit(incre_feature, incre_label)
    clear_mermory(test_b_path, test_b, probs, pu_test_b_seg_1, incre, incre_feature, incre_label)

    return _clf, test_b_seg_2, score_seg_1

def get_score(clf, test_b_seg_2, score_seg_1, score_path):

    #Get score
    start = time.time()
    print("\n# GET RESULT")
    probs = clf.predict_proba(test_b_seg_2.iloc[:,2:])
    score_seg_2 = pd.DataFrame(test_b_seg_2["id"]).assign(score = probs[:,1])
    score = score_seg_1.append(score_seg_2).sort_index()
    score.to_csv(log_path + "last_3_days/score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute))
    print("\n# Score saved in {}".format(score_path))
    clear_mermory(probs, score_seg_2, score)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

def main():

    clf, train, roc_init = init_train(train_path)
    #_, pu_train, roc_pu = positive_unlabel(clf, test_a_path, train, pu_thresh_a)
    #clf, test_b_seg_2, score_seg_1 = part_fit(clf, test_b_path, pu_train, pu_thresh_b)
    get_score(clf, test_b_seg_2, score_seg_1, score_path)

    log_parmas(clf, log_path + "log_{}h.csv".format(now.hour),
            roc_init = roc_init, roc_pu = roc_pu,
            pu_thresh_a = pu_thresh_a, pu_thresh_b = pu_thresh_b )

if __name__ == '__main__':
    main()
