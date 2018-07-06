import pandas as pd
from lib.data_processing import *
from lib.model_performance import *
import sys

"""
## DEBUG:
train_path = "data/train.csv"
_train_data = pd.read_csv(train_path)
_train_data = custom_imputation(_train_data)
_train_data = _train_data.replace({"label" : -1}, value = 1)
_train_data.info(memory_usage='deep')

_train_data.iloc[:,1:] = _train_data.iloc[:,1:].astype('int')
_train_data.iloc[:,0] = _train_data.iloc[:,0].astype('category')

_train_data.info(memory_usage='deep')


for dtype in ['float','int','object']:
    selected_dtype = _train_data.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
"""
def test():
