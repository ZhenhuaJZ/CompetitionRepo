import pandas as pd
from lib.data_processing import *
from lib.model_performance import *
import sys

#data_1 = pd.DataFrame([[1,2,3,4],[2,5,6,7]], columns=['date','f1','f2','f3'])
#data_2 = pd.DataFrame([[3,2,3,6],[4,5,6,7]],columns=['date','f1','f2','f3'])

data_1 = "_train_data_test.csv"
data_2 = "pu_black_data_test.csv"

df1 = pd.read_csv(data_1)
df2 = pd.read_csv(data_2)

merged_file = file_merge(df1, df2, "date")
print(merged_file)
