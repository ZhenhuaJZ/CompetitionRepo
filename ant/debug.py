import pandas as pd
from lib.data_processing import *
from lib.model_performance import *
import sys


# DEBUG: for file merge
data_1 = "_train_data_test.csv"
data_2 = "pu_black_data_test.csv"

df1 = pd.read_csv(data_1)
df2 = pd.read_csv(data_2)

merged_file = file_merge(df1, df2, "date")
print(merged_file)

## DEBUG: for call fit and add call DataFrame
