import pandas as pd
from xgboost import XGBClassifier
from lib.data_processing import *
from lib.model_performance import *
from core_model import pu_labeling, partical_fit, cv_fold, grid_search_roc
from sklearn.externals import joblib
from sklearn.base import clone
import time, sys, datetime
from lib.tool import *
from stack import *
now = datetime.datetime.now()

score_path = "log/last_1_day/{}d_{}h_{}m/".format(now.day, now.hour, now.minute)
params_path = "log/last_1_day/log_{}h.csv".format(now.hour)

train_path = "log/last_1_day/8d_3h_45m/layer1_train_3:45.csv"
test_b_path = "log/last_1_day/8d_3h_45m/layer1_test_3:45.csv"
label_path = "data/stack_train_best.csv"
score_id_path = "data/test_b.csv"

def main():

    print("\n# Make dirs in {}".format(score_path))
    print("\n# Train_path : {}".format(train_path))
    os.makedirs(score_path)
    sys.stdout = Logger(score_path + "log_{}d_{}h_{}m.txt".format(now.day, now.hour, now.minute))

    #Read data
    train = pd.read_csv(train_path, header = None, low_memory = False)
    test_b = pd.read_csv(test_b_path, header = None)
    label = pd.read_csv(label_path, low_memory = False)
    score_id = pd.read_csv(score_id_path)

    #Read second layer
    probs = read_saved_layer(train, test_b, label)

    #Save score
    score = pd.DataFrame(score_id["id"]).assign(score = probs)
    _score_path = score_path  + "stacking_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
    score.to_csv(_score_path, index = None, float_format = "%.9f")
    print("\n# Stacking Score saved in {}".format(_score_path))

if __name__ == '__main__':
    main()
