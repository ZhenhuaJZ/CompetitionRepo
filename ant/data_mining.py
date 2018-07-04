import pandas as pd
from xgboost import XGBClassifier
from lib.data_processing import *
from lib.model_performance import *
from core_model import positive_unlabel_learning, partical_fit
import time, sys

train_path = "data/_train_data.csv"
test_b_path = "data/test_b.csv"
test_a_path = "data/test_a.csv"
validation_path = "data/test_offline.csv"

#train = pd.read_csv(train_path)
#validation = pd.read_csv(validation_path)
#test_a = pd.read_csv(test_a_path)
#test_b = pd.read_csv(test_b_path)

def main():

    start = time.time()
    clf = XGBClassifier(max_depth = 4, n_estimators = 480, subsample = 0.8, gamma = 0,
                    min_child_weight = 1, scale_pos_weight = 1,
                    colsample_bytree = 0.8, learning_rate = 0.07, n_jobs = -1),
    #Train
    train = pd.read_csv(train_path)
    feature, label = split_train_label(train)
    clf = clf.fit(feature, label)
    clear_mermory(feature, label)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

    #PU
    test_a = pd.read_csv(test_a_path)
    pu_black = positive_unlabel_learning(clf, test_a, 0.5)
    pu_train = file_merge(train, pu_black, "date")
    clear_mermory(train, pu_black)
    pu_feature, pu_label = split_train_label(pu_train)
    clf = clf.fit(pu_feature, pu_label)
    clear_mermory(pu_feature, pu_label)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
    
    #Eval PU
    validation = pd.read_csv(validation_path)
    val_feature, val_label = split_train_label(validation)
    val_probs = clf.predict_proba(val_feature)
    ant_score = offline_model_performance(val_label, val_probs[:,1])
    clear_mermory(val_feature, val_label, validation)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))
    sys.exit()

    #Partical_fit




    #probs = clf.predict_proba(test_online)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))



if __name__ == '__main__':
    main()
