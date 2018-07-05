import pandas as pd
from xgboost import XGBClassifier
from lib.data_processing import *
from lib.model_performance import *
from core_model import positive_unlabel_learning, partical_fit, cv_fold
import time, sys, datetime
now = datetime.datetime.now()

score_path = "log/last_3_days/{}d_{}h_{}m/".format(now.day, now.hour, now.minute)
params_path = "log/last_3_days/log_{}h.csv".format(now.hour)

train_path = "data/_train_data.csv"
test_b_path = "data/test_b.csv"
test_a_path = "data/test_a.csv"
validation_path = "data/_test_offline.csv"


pu_thresh_a = 0.80 #PU threshold for testa
pu_thresh_b = 0.80 #PU threshold for testb
partial_rate = 0.4
################################################################################
## DEBUG:
debug = False
################################################################################

def init_train(clf, eval = True, save_score = True):
    over_sampling = False

    start = time.time()

    #Train
    print("\n# Start Traing")
    print("\n# {}".format(clf))
    train = pd.read_csv(train_path)
    if over_sampling:
        train = over_sampling(train, 0.2)

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

    if save_score:

        test_b = pd.read_csv(test_b_path)
        probs = clf.predict_proba(test_b.iloc[:,2:])
        score = pd.DataFrame(test_b["id"]).assign(score = probs[:,1])
        _score_path = score_path + "inti_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Score saved in {}".format(_score_path))

    return clf, train, roc

def positive_unlabel(clf, train, pu_thresh_a, eval = True, save_score = True):
    #PU
    start = time.time()
    print("\n# START PU - TESTA , PU_thresh_a = {}".format(pu_thresh_a))
    test_a = pd.read_csv(test_a_path)
    pu_black = positive_unlabel_learning(clf, test_a, pu_thresh_a)
    _train = file_merge(train, pu_black, "date")
    _feature, _label = split_train_label(_train)
    clf.fit(_feature, _label)
    clear_mermory(_feature, _label, train, pu_black, test_a_path, test_a)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if eval:

        print("\n# EVAL PU")
        validation_path = "data/_test_offline.csv"
        validation = pd.read_csv(validation_path)
        val_feature, val_label = split_train_label(validation)
        val_probs = clf.predict_proba(val_feature)
        roc = offline_model_performance(val_label, val_probs[:,1])
        clear_mermory(val_feature, val_label, validation, validation_path, val_probs)

    if save_score:

        test_b = pd.read_csv(test_b_path)
        probs = clf.predict_proba(test_b.iloc[:,2:])
        score = pd.DataFrame(test_b["id"]).assign(score = probs[:,1])
        _score_path = score_path  + "pu_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Score saved in {}".format(_score_path))

    return _train, roc

def validation_black(clf, train, save_score = True):
    #Feed validation black label Back
    print("\n# Feed Validation Black Label Back")
    start = time.time()
    validation_path = "data/_test_offline.csv"
    validation = pd.read_csv(validation_path)
    validation_black = validation.loc[validation["label"] == 1]
    print("\n# Found <{}> black instances".format(len(validation_black)))
    _train = file_merge(train, validation_black, "date")
    _feature, _label = split_train_label(_train)
    clf.fit(_feature, _label)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if save_score:

        test_b = pd.read_csv(test_b_path)
        probs = clf.predict_proba(test_b.iloc[:,2:])
        score = pd.DataFrame(test_b["id"]).assign(score = probs[:,1])
        _score_path = score_path  + "val_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Score saved in {}".format(_score_path))
    return  _train

def part_fit(clf, train, partial_rate, pu_thresh_b, eval = True, save_score = True):
    #Partical_Fit
    start = time.time()
    print("\n# PART FIT TESTB, PU_thresh_b = {}, Partial_Rate = {}".format(pu_thresh_b, partial_rate))
    test_b = pd.read_csv(test_b_path)
    test_b_seg_1, test_b_seg_2 = partical_fit(test_b, partial_rate, "date")
    _feature, _label = split_train_label(train)
    clf.fit(_feature, _label)
    probs = clf.predict_proba(test_b_seg_1.iloc[:,2:])
    score_seg_1 = pd.DataFrame(test_b_seg_1["id"]).assign(score = probs[:,1])
    test_b_seg_1_black = positive_unlabel_learning(clf, test_b_seg_1, pu_thresh_b) #pu threshold
    _train = file_merge(train, test_b_seg_1_black, "date")
    _feature, _label = split_train_label(_train)
    clf.fit(_feature, _label)
    clear_mermory(test_b_path, test_b, probs, test_b_seg_1_black, _train, _feature, _label, train)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if save_score:
        #Get score
        probs = clf.predict_proba(test_b_seg_2.iloc[:,2:])
        score_seg_2 = pd.DataFrame(test_b_seg_2["id"]).assign(score = probs[:,1])
        score = score_seg_1.append(score_seg_2).sort_index()
        _score_path = score_path + "part_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Score saved in {}".format(_score_path))
        clear_mermory(probs, score_seg_2, score)

    if eval:
        #CV -5 Folds
        slice_interval = [[20170905, 20170916], [20170917, 20170925], [20170926, 20171005],[20171006,20171015],[20171015,20171025]]
        roc = cv_fold(clf, _train, slice_interval)

    return _train, roc

def pu_a():
    _clf = XGBClassifier(max_depth = 4, n_estimators = 4, subsample = 0.8, gamma = 0.1,
                    min_child_weight = 1, scale_pos_weight = 1,
                    colsample_bytree = 0.8, learning_rate = 0.08, n_jobs = -1)

    clf, train, roc_init = init_train(_clf)
    pu_train, roc_pu = positive_unlabel(clf, train, pu_thresh_a)
    # TODO: FINE TUNING CLF2
    pu_train = validation_black(_clf, pu_train)
    log_parmas(_clf, params_path, score_path = score_path, roc_init = round(roc_init,6),
                roc_pu = round(roc_pu,6),pu_thresh_a = pu_thresh_a)

    return pu_train

def pu_b(pu_train, pu_test_b = True):
    _clf = XGBClassifier(max_depth = 4, n_estimators = 5, subsample = 0.8, gamma = 0.1,
                    min_child_weight = 1, scale_pos_weight = 1,
                    colsample_bytree = 0.8, learning_rate = 0.08, n_jobs = -1)

    if pu_test_b:
        roc_part = part_fit(_clf, pu_train, partial_rate, pu_thresh_b, eval = True)
        log_parmas(_clf, params_path, score_path = score_path, roc_part = round(roc_part,6),
                    pu_thresh_b = pu_thresh_b,  partial_rate = partial_rate)

    return

def main():
    os.makedirs(score_path)
    print("\n# Make dirs in {}".format(score_path))

    pu_train = pu_a()
    pu_b(pu_train, pu_test_b = True)

if __name__ == '__main__':
    main()

# clf, train, roc_init = init_train(save_score = True)
# pu_train, roc_pu = positive_unlabel(clf, train, pu_thresh_a, save_score = True)
# part_train, roc_part = part_fit(clf, pu_train, partial_rate, pu_thresh_b, save_score = True)
# validation_black(clf, part_train, save_score = True)
