import pandas as pd
from xgboost import XGBClassifier
from lib.data_processing import *
from lib.model_performance import *
from core_model import positive_unlabel_learning, partical_fit, cv_fold
from sklearn.externals import joblib
import time, sys, datetime
now = datetime.datetime.now()

score_path = "log/last_3_days/{}d_{}h_{}m/".format(now.day, now.hour, now.minute)
params_path = "log/last_3_days/log_{}h.csv".format(now.hour)

train_path = "data/_train_data.csv"
test_b_path = "data/test_b.csv"
test_a_path = "data/test_a.csv"
validation_path = "data/_test_offline.csv"
filename = "5d_17h_9m"


pu_thresh_a_range = [0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87]

def load_model():

    model_path = "log/last_3_days/" + filename + "/inti_model.pkl"
    test_a = pd.read_csv(test_a_path)
    train = pd.read_csv(train_path)

    clf = joblib.load(model_path)
    for pu_thresh_a in pu_thresh_a_range:

        print("\n# Tuning {}".format(pu_thresh_a) )
        pu_black = positive_unlabel_learning(clf, test_a, pu_thresh_a)
        _train = file_merge(train, pu_black, "date")
        _feature, _label = split_train_label(_train)
        #to do fine tunning
        clf.fit(_feature, _label)

        print("\n# EVAL PU")
        validation_path = "data/_test_offline.csv"
        validation = pd.read_csv(validation_path)
        val_feature, val_label = split_train_label(validation)
        val_probs = clf.predict_proba(val_feature)
        roc = offline_model_performance(val_label, val_probs[:,1])

def main():
    load_model()

if __name__ == '__main__':
    main()
