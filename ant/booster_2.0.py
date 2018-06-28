from xgboost import XGBClassifier
from pipeline import *
now = datetime.datetime.now()

# #####################File path#########################################
log_path = "log/date_{}/SM_{}:{}/".format(now.day,now.hour,now.minute)
params_path = log_path + "params/"
score_path = log_path + "score/"
model_path = log_path + "model/"
as_path = "lib/answer_sheet.csv"


train_path = "data/train.csv" #train_heatmap , train_mode_fill, train,
test_path = "data/test_b.csv" #test_a_heatmap, test_a_mode_fill, test_b
fillna_value = 0

_train_data = pd.read_csv(train_path)
_test_online = pd.read_csv(test_path)
print("check out ", len(_train_data))
_train_data, _test_online = custom_imputation(_train_data, _test_online, fillna_value)
_train_data.loc[_train_data["label"] == -1] = 1
print("check out ", len(_train_data))

_train, _test_offline =  test_train_split_by_date(_train_data, 20170905, 20170915)

_train = _train.iloc[:,3:]
_label = _train.iloc[:,1]
_test_offline = _test_offline.iloc[:,3:]
_test_offline_labels = _test_offline.iloc[:,1]

xgb = XGBClassifier(max_depth = 3, n_estimators = 5, subsample = 0.9,
					colsample_bytree = 0.8, learning_rate = 0.1)

xgb = xgb.fit(_train, _labels)
probs = xgb.predict_proba(_test_online)
joblib.dump(best_est, model_path + "{}.pkl".format(method))

offline_score = offline_model_performance(xgb, _test_offline, _test_offline_labels)

save_score(probs[:,1])


def main():
	os.makedirs(log_path)
	os.makedirs(score_path)
	os.makedirs(params_path)
	os.makedirs(model_path)

if __name__ == '__main__':
    main()
