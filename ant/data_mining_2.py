import pandas as pd
from xgboost import XGBClassifier
from lib.data_processing import *
from lib.model_performance import *
from core_model import *
from sklearn.externals import joblib
from sklearn.base import clone
import time, sys, datetime
from lib.tool import *
from stack import *
now = datetime.datetime.now()

score_path = "log/last_1_day/{}d_{}h_{}m/".format(now.day, now.hour, now.minute)
params_path = "log/last_1_day/log_{}h.csv".format(now.hour)

#train_path = "data/stack_train_best.csv"
train_path = "data/train_normal_unlabel_float.csv"  #train_normal_un.csv, train_float64.csv, train_normal_unlabel_float
test_set_path = "data/test_normal_unlabel_float.csv" #validation_normal_un.csv, validation_float64, test_normal_unlabel_float
test_b_path = "data/test_b.csv"
test_a_path = "data/test_a.csv"
corr_data = "data/corr_data.npy"

model_path =  None
stacking = True
over_samp = True
pu_test_b = True

over_samp_ratio = 0.030 # 0.06 add 808 to train
thresh_a = 0.90 #PU threshold for testa
thresh_b = 0.95 #PU threshold for testb
seg_date = 20180215
params = None
feature_drops = np.load(corr_data).tolist()

xgb_a = XGBClassifier(max_depth = 4, n_estimators = 480, subsample = 0.8, gamma = 0,
                min_child_weight = 1, scale_pos_weight = 1,
                colsample_bytree = 0.8, learning_rate = 0.07, n_jobs = -1)

xgb_b = XGBClassifier(max_depth = 4, n_estimators = 460, subsample = 0.8, gamma = 0,
                min_child_weight = 1, scale_pos_weight = 1,
                colsample_bytree = 0.8, learning_rate = 0.06, n_jobs = -1)

def positive_unlabel_learning(clf, data_path, train, thresh, prefix = "pu"):
    start = time.time()
    roc = "n/a"
    unlabel = pd.read_csv(data_path)
    if len(feature_drops) != 0:
        unlabel = unlabel.drop(feature_drops, axis = 1)
    black = pu_labeling(clf, unlabel, thresh)
    _train = file_merge(train, black, "date")
    _feature, _label = split_train_label(_train)
    clf.fit(_feature, _label)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    return clf, _train

def init_train(clf, store_score = True, save_model = False, model_path = None, params = None):

    start = time.time()
    print("\n# Start Traing")
    print("\n# Init classifier : \n{}".format(clf))
    train = pd.read_csv(train_path)

    if len(feature_drops) != 0:
        print("\n Feature selection droped features {}".format(feature_drops))
        train = train.drop(feature_drops, axis = 1)

    if over_samp:
        train = SMOTE_sampling(train, over_samp_ratio)

    if params != None:
        #test_set = pd.read_csv(test_set_path)
        #clf = grid_search_roc(clf, train, test_set, params)
        # TODO:
        print("\n# invalid grid search")

    feature, label = split_train_label(train)

    if model_path != None :
        clf = joblib.load(model_path)
        print("\n# Load Model from {}".format(model_path))

    clf.fit(feature, label)

    if save_model and dump_model == None:
        joblib.dump(clf, score_path + "inti_model.pkl")
        print("\n# Model dumped")
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    if store_score:
        save_score(clf, test_b_path, score_path, feature_drops, prefix = "inti")

    return clf, train

def part_fit(clf, train, seg_date, pu_thresh_b, store_score = True):
    #Partical_Fit
    roc = 0
    start = time.time()
    print("\n# PART FIT TESTB, PU_thresh_b = {}, Seg_Date = {}".format(pu_thresh_b, seg_date))
    print("\n# Part_Fit Classifier {} : \n".format(clf))
    test_b = pd.read_csv(test_b_path)
    #Drop features
    if len(feature_drops) != 0:
        test_b = test_b.drop(feature_drops, axis = 1)
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

    return clf, _train

def pu_a(clf):

    _clf, _train = init_train(clf,  model_path = model_path, params = params)
    #roc_val, roc_test = evaluation(_clf, test_set_path, _train)
    #print("\n# Tuning init parmas")

    print("\n# START PU - TESTA , PU_thresh_A = {}".format(thresh_a))
    _clf, _train = positive_unlabel_learning(_clf, test_a_path, _train, thresh_a, prefix = "pua")
    return  _clf, _train

def pu_b(clf, train):

    print("\n# START PU - TESTB , PU_thresh_B = {}".format(thresh_b))
    _clf, _train = part_fit(clf, train, seg_date, thresh_b)
    return _clf, _train

def main():
    os.makedirs(score_path)
    sys.stdout = Logger(score_path + "log_{}d_{}h_{}m.txt".format(now.day, now.hour, now.minute))
    print("\n# Make dirs in {}".format(score_path))
    print("\n# Train_path : {}".format(train_path))
    print("\n# Test_set_path : {}".format(test_set_path))

    _clf, _train = pu_a(xgb_a)
    #roc_val, roc_test = evaluation(_clf, test_set_path, _train)

    if pu_test_b:
        _clf, _train = pu_b(xgb_b, _train)

        test_set = pd.read_csv(test_set_path)

        if len(feature_drops) != 0:
            test_set = test_set.drop(feature_drops, axis = 1)

        roc_val, roc_test = evaluation(_clf, test_set, _train)
    #############################Stacking#######################################

    if stacking:

        _train = pd.read_csv(train_path)
        test_b = pd.read_csv(test_b_path)

        if len(feature_drops) != 0:
            _train = _train.drop(feature_drops, axis = 1)
            test_b = test_b.drop(feature_drops, axis = 1)

        #_train.iloc[:,1:] = _train.iloc[:,1:].astype('int32')
        #_train.iloc[:,0] = _train.iloc[:,0].astype('category')
        #test_b.iloc[:,1:] = test_b.iloc[:,1:].astype('int32')
        #test_b.iloc[:,0] = test_b.iloc[:,0].astype('category')

        probs = two_layer_stacking(_train, test_b)

        score = pd.DataFrame(test_b["id"]).assign(score = probs)
        _score_path = score_path  + "stacking_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Stacking Score saved in {}".format(_score_path))

if __name__ == '__main__':
    main()
