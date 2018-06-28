from hparams import *
from pipeline import *

#data path
train_path = "data/train.csv" #train_heatmap , train_mode_fill, train,
test_path = "data/test_b.csv" #test_a_heatmap, test_a_mode_fill, test_b

################################################################################

# #method_1_describ = ["MinMaxScaler", "Kbest", "Xgboost"]
# #method_2_describ = ["StandardScaler", "Tree-Base Importance Feature", "Xgboost"]

################################################################################
method = "method_2"
fillna_value = 0

if __name__ == '__main__':
    main(method, train_path, test_path, fillna_value)
