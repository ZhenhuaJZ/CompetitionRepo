import pandas as pd
from lib.data_processing import *
from lib.model_performance import *
import time

train_path = "data/_train_data.csv"
test_b_path = "data/test_b.csv"
test_a_path = "data/test_a.csv"
validation_path = "data/test_offline.csv"

#feature Engineer
train = pd.read_csv(train_path)
#validation = pd.read_csv(validation_path)
#test_a = pd.read_csv(test_a_path)
#test_b = pd.read_csv(test_b_path)

def main():
    
    start = time.time()
	clf = XGBClassifier(max_depth = 4, n_estimators = 480, subsample = 0.8, gamma = 0,
                        min_child_weight = 1, scale_pos_weight = 1,
	                    colsample_bytree = 0.8, learning_rate = 0.07, n_jobs = -1),

    train = pd.read_csv(train_path)
    feature, label = split_train_label(train)
    clf = clf.fit(feature, label)
    clear_mermory(_train, _labels)
    probs = clf.predict_proba(test_online)
    print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))



if __name__ == '__main__':
    main()
