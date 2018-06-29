from hparams import *
from pipeline import *
train_path = "data/train.csv"
test_path = "data/test_b.csv"

################################################################################

# #method_1_describ = ["MinMaxScaler", "Kbest", "Xgboost"]
# #method_2_describ = ["StandardScaler", "Tree-Base Importance Feature", "Xgboost"]

################################################################################
method = "method_3"
fillna_value = 0

if __name__ == '__main__':
    main(method, train_path, test_path, fillna_value)
