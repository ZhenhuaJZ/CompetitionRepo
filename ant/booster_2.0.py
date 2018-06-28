from xgboost import XGBClassifier
from data_processing import save_score, test_train_split_by_date
from model_performance import offline_model_performance
now = datetime.datetime.now()

# #####################File path################################################
log_path = "log/date_{}/SM_{}:{}/".format(now.day,now.hour,now.minute)
params_path = log_path + "params/"
score_path = log_path + "score/"
model_path = log_path + "model/"

# #####################Data path################################################
train_path = "data/train.csv" #train_heatmap , train_mode_fill, train,
test_path = "data/test_b.csv" #test_a_heatmap, test_a_mode_fill, test_b
fillna_value = 0

_train_data = pd.read_csv(train_path)
_test_online = pd.read_csv(test_path)

_train_data, _test_online = custom_imputation(_train_data, _test_online, fillna_value)
#change -1 label to 1
_train_data.loc[_train_data["label"] == -1] = 1
#Split train and offine test
_train, _test_offline =  test_train_split_by_date(_train_data, 20171020, 20171031)
#train data
_train = _train.iloc[:,3:]
_labels = _train.iloc[:,1]
#online & offline data
_test_online = _test_online.iloc[:,2:]
_test_offline_feature = _test_offline.iloc[:,3:]
_test_offline_labels = _test_offline.iloc[:,1]

xgb = XGBClassifier(max_depth = 3, n_estimators = 5, subsample = 0.9,
					colsample_bytree = 0.8, learning_rate = 0.1)
xgb = xgb.fit(_train, _labels)
probs = xgb.predict_proba(_test_online)
joblib.dump(xgb, model_path + "{}.pkl".format(method))
offline_score = offline_model_performance(xgb, _test_offline_feature, _test_offline_labels)
save_score(probs[:,1], score_path)

def main():
	os.makedirs(log_path)
	os.makedirs(score_path)
	os.makedirs(params_path)
	os.makedirs(model_path)

if __name__ == '__main__':
    main()
