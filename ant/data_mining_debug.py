import pandas as pd
from xgboost import XGBClassifier
from lib.data_processing import *
from lib.model_performance import *
from core_model import pu_labeling, partical_fit, cv_fold, grid_search_roc
from sklearn.externals import joblib
from sklearn.base import clone
import time, sys, datetime
from lib.tool import *
from stack_debug import *
now = datetime.datetime.now()

score_path = "log/last_1_day/{}d_{}h_{}m/".format(now.day, now.hour, now.minute)
params_path = "log/last_1_day/log_{}h.csv".format(now.hour)

train_path = "data/stack_train_best.csv"
#train_path = "data/train_float64.csv"  #train_normal_un.csv, train_float64.csv, train_normal_unlabel_float
validation_path = "data/validation_float64.csv" #validation_normal_un.csv, validation_float64, test_normal_unlabel_float
test_b_path = "data/test_b.csv"
test_a_path = "data/test_a.csv"
model_name = None #"6d_23h_10m" #best score model
corr_data = "data/corr_data.npy"

stacking = True
over_samp = False
over_samp_ratio = 0.019 # 0.06 add 808 to train
pu_thresh_a = 0.486 #PU threshold for testa
pu_test_b = False
pu_thresh_b = 0.88 #PU threshold for testb
seg_date = 20180215
params =  None

def positive_unlabel_learning(clf, data_path, train, thresh, eval = True, save_score = True, prefix = "pu"):

    start = time.time()
    roc = "n/a"
    unlabel = pd.read_csv(data_path)
    black = pu_labeling(clf, unlabel, thresh)
    _train = file_merge(train, black, "date")
    feature, label = split_train_label(_train)
    #clf.set_params(learning_rate = 0.06, n_estimators = 460)
    print("\n# f ine_tune : 1 :\n", clf)
    clf.fit(feature, label)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if save_score:

        test_b = pd.read_csv(test_b_path)
        probs = clf.predict_proba(test_b.iloc[:,2:])
        score = pd.DataFrame(test_b["id"]).assign(score = probs[:,1])
        _score_path = score_path  + "{}_score_{}d_{}h_{}m.csv".format(prefix, now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Score saved in {}".format(_score_path))

    if eval:

        print("\n# EVAL PU")
        validation = pd.read_csv(validation_path)
        val_feature, val_label = split_train_label(validation)
        val_probs = clf.predict_proba(val_feature)
        roc = offline_model_performance(val_label, val_probs[:,1])
        clear_mermory(val_feature, val_label, validation, validation_path, val_probs)

    return clf, _train, roc

def init_train(clf, eval = True, save_score = True, save_model = False, params = None, dump_model = None):

    start = time.time()
    #Train
    print("\n# Start Traing")
    print("\n# {}".format(clf))
    train = pd.read_csv(train_path)

    if over_samp:
        train = over_sampling(train, over_samp_ratio)

    if params != None:
        validation = pd.read_csv(validation_path)
        clf = grid_search_roc(clf, train, validation, params)
    feature, label = split_train_label(train)

    if dump_model != None :
        model_path =  "log/last_3_days/" + dump_model + "/inti_model.pkl"
        print("\n# Load Model from {}".format(dump_model))
        clf = joblib.load(model_path)

    clf.fit(feature, label)

    if save_model and dump_model == None:
        joblib.dump(clf, score_path + "inti_model.pkl")
        print("\n# Model dumped")
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if eval:

        print("\n# EVAL INIT CLASSIFIER")
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

def validation_black(clf, train, eval = True, save_score = True, save_model = True):
    #Feed validation black label Back
    roc = 0
    print("\n# Feed Validation Black Label Back")
    start = time.time()
    validation = pd.read_csv(validation_path)
    validation_black = validation.loc[validation["label"] == 1]
    print("\n# Found <{}> black instances".format(len(validation_black)))
    _train = file_merge(train, validation_black, "date")
    _feature, _label = split_train_label(_train)
    clf.fit(_feature, _label)
    if save_model:
        joblib.dump(clf, score_path + "val_model.pkl")
        print("\n# Model dumped")
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if save_score:

        test_b = pd.read_csv(test_b_path)
        probs = clf.predict_proba(test_b.iloc[:,2:])
        score = pd.DataFrame(test_b["id"]).assign(score = probs[:,1])
        _score_path = score_path  + "val_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Score saved in {}".format(_score_path))

    if eval:
        #CV -5 Folds seg by date
        _day = []
        interval = int(len(_train["date"])/5)
        for i in range(6):
            _day.append(_train["date"].iloc[interval*i])
        slice_interval = [[_day[0], _day[1]], [_day[1]+1, _day[2]], [_day[2]+1, _day[3]],[_day[3]+1,_day[4]],[_day[4]+1, _day[5]]]
        roc = cv_fold(clf, _train, slice_interval)
        print("\n# Val 5 : CV5 score {}".format(roc))

    return  _train, roc

def part_fit(clf, train, seg_date, pu_thresh_b, eval = True, save_score = True):
    #Partical_Fit
    roc = 0
    start = time.time()
    print("\n# PART FIT TESTB, PU_thresh_b = {}, Seg_Date = {}".format(pu_thresh_b, seg_date))
    print("\n# {}".format(clf))
    test_b = pd.read_csv(test_b_path)
    test_b_seg_1, test_b_seg_2 = partical_fit(test_b, seg_date, "date")
    _feature, _label = split_train_label(train)
    clf.fit(_feature, _label)
    probs = clf.predict_proba(test_b_seg_1.iloc[:,2:])
    score_seg_1 = pd.DataFrame(test_b_seg_1["id"]).assign(score = probs[:,1])
    test_b_seg_1_black = pu_labeling(clf, test_b_seg_1, pu_thresh_b) #pu threshold
    _train = file_merge(train, test_b_seg_1_black, "date")
    _feature, _label = split_train_label(_train)
    clf.fit(_feature, _label)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if save_score:
        #Get score
        probs = clf.predict_proba(test_b_seg_2.iloc[:,2:])
        score_seg_2 = pd.DataFrame(test_b_seg_2["id"]).assign(score = probs[:,1])
        score = score_seg_1.append(score_seg_2).sort_index()
        _score_path = score_path + "part_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Score saved in {}".format(_score_path))

    if eval:
        #CV -5 Folds seg by date
        _day = []
        interval = int(len(_train["date"])/5)
        for i in range(6):
            _day.append(_train["date"].iloc[interval*i])
        slice_interval = [[_day[0], _day[1]], [_day[1]+1, _day[2]], [_day[2]+1, _day[3]],[_day[3]+1,_day[4]],[_day[4]+1, _day[5]]]
        roc = cv_fold(clf, _train, slice_interval)
        print("\n# Partb: CV5 score {}".format(roc))
        print("\n# Partb: CV5 score {}".format(roc))
    return roc

def pu_a():

    _clf = XGBClassifier(max_depth = 4, n_estimators = 480, subsample = 0.8, gamma = 0.1,
                    min_child_weight = 1, scale_pos_weight = 1,
                    colsample_bytree = 0.8, learning_rate = 0.07, n_jobs = -1)

    clf, train, roc_init = init_train(_clf, params = params, dump_model = model_name)
    #print("\n# START PU - UNLABEL , PU_thresh_unlabel = {}".format(pu_unlabel))
    #clf, train, roc_unlabel = positive_unlabel_learning(clf, unlabel_path, train, pu_unlabel, prefix = "un_pu")

    print("\n# START PU - TESTA , PU_thresh_a = {}".format(pu_thresh_a))
    _, train, roc_pua = positive_unlabel_learning(clf, test_a_path, train, pu_thresh_a, prefix = "pua")

    # TODO: Fine tunning
    #_clf.set_params(n_estimators = 350, learning_rate = 0.06)
    print("\n# fine_tune : 2 : \n", _clf)

    _train = validation_black(_clf, train)
    _train.to_csv("data/after_valdation_data.csv", index =None)

    log_parmas(_clf, params_path, roc_init = round(roc_init,6),#roc_unlabel = round(roc_unlabel,6), pu_unlabel = pu_unlabel,
                roc_pua = round(roc_pua,6), pu_thresh_a = pu_thresh_a, score_path = score_path, over_samp = over_samp,
                over_samp_ratio = over_samp_ratio, bst_clf = clf)

    #return
    return _train

def pu_b(train, pu_test_b, eval):
    _clf = XGBClassifier(max_depth = 4, n_estimators = 480, subsample = 0.8, gamma = 0.1,
                    min_child_weight = 1, scale_pos_weight = 1,
                    colsample_bytree = 0.8, learning_rate = 0.06, n_jobs = -1)

    if pu_test_b:
        roc_part = part_fit(_clf, train, seg_date, pu_thresh_b, eval)
        log_parmas(_clf, params_path, roc_part = round(roc_part,6), pu_thresh_b = pu_thresh_b, seg_date = seg_date, score_path = score_path)

    return

def main():
    print("\n# Make dirs in {}".format(score_path))
    print("\n# Train_path : {}".format(train_path))
    print("\n# Validation_path : {}".format(validation_path))
    os.makedirs(score_path)
    sys.stdout = Logger(score_path + "log_{}d_{}h_{}m.txt".format(now.day, now.hour, now.minute))
    """
    train = pu_a()
    #train.to_csv("data/stack_train_best.csv", index = None)
    pu_b(train, pu_test_b, eval = True)
    """

    if stacking:

        train = pd.read_csv(train_path, low_memory = False)
        test_b = pd.read_csv(test_b_path)
        probs = two_layer_stacking(train, test_b)

        score = pd.DataFrame(test_b["id"]).assign(score = probs)
        _score_path = score_path  + "stacking_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Stacking Score saved in {}".format(_score_path))

if __name__ == '__main__':
    main()
