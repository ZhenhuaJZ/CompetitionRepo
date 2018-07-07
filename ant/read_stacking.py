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

test_b_path = "data/test_b.csv"
train_path = "data/stack_train_best.csv"

_train = pd.read_csv(train_path)
test_b = pd.read_csv(test_b_path)
probs = two_layer_stacking(_train, test_b)

score = pd.DataFrame(test_b["id"]).assign(score = probs)
_score_path = score_path  + "stacking_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
score.to_csv(_score_path, index = None, float_format = "%.9f")
print("\n# Stacking Score saved in {}".format(_score_path))
