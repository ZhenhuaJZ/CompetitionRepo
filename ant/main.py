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

# #########################Main data########################################
_train_data = pd.read_csv(train_path)
_test_online = pd.read_csv(test_path)
_train_data, _test_online = custom_imputation(_train_data, _test_online, fillna_value)

#_train_data = _train_data[(_train_data.label==0)|(_train_data.label==1)]
#_train,  _test_offline = test_train_split_by_date(_train_data, 20170905, 20170910)

_train = _train_data[(_train_data.label==0)|(_train_data.label==1)] #for test purpose

_test_offline = _train.iloc[1001:3200,3:] # for test purpose
_test_offline_labels = _train.iloc[1001:3200,1] #for test purpose

_train = _train.iloc[:1000,3:]
_labels = _train.iloc[:1000,1]
#_test_offline = _test_offline.iloc[:,3:]
#_test_offline_labels = _test_offline.iloc[:,1]
_test_online = _test_online.iloc[:,2:]

if __name__ == '__main__':


    #Split data
    #_train, _test_offline, _labels, _test_offline_label = train_test_split(_train, labels, test_size = test_size, random_state = 42, shuffle = False)

    main(method, _train, _labels, _test_online, _test_offline, _test_offline_labels, fillna_value)
