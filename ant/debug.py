import pandas as pd
from lib.data_processing import *
from lib.model_performance import *
import sys

## DEBUG: 
train_path = "data/train.csv"
_train_data = pd.read_csv(train_path)
_train_data = custom_imputation(_train_data)
_train_data = _train_data.replace({"label" : -1}, value = 1)
_train_data.info(memory_usage='deep')
