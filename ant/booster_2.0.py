import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from lib.data_processing import *
from lib.model_performance import offline_model_performance
import datetime, time
now = datetime.datetime.now()

log_path = "log/date_{}/{}:{}_SM/".format(now.day,now.hour,now.minute)
params_path = log_path + "params/"
score_path = log_path + "score/"
model_path = log_path + "model/"
creat_project_dirs(log_path, params_path, score_path, model_path)
# #####################Data path################################################
train_path = "data/train.csv" #train_heatmap , train_mode_fill, train,
test_path = "data/test_b.csv" #test_a_heatmap, test_a_mode_fill, test_b
test_a_path = "data/test_a.csv"
fillna_value = 0

_train_data = pd.read_csv(train_path)
_test_online = pd.read_csv(test_path)
#Using test_a as
_test_a = pd.read_csv(test_a_path)

_train_data, _test_online, _test_a = custom_imputation(_train_data, _test_online, fillna_value, _test_a)
#change -1 label to 1
_train_data.loc[_train_data["label"] == -1] = 1
#Split train and offine test
_train_data, _test_offline =  test_train_split_by_date(_train_data, 20171010, 20171020, params_path)
#train data
"""
_train = _train_data.iloc[:,3:]
_labels = _train_data.iloc[:,1]
"""
_train, _labels = split_train_label(_train_data, cache = True)
#online & offline data
_test_online = _test_online.iloc[:,2:]

"""
_test_offline_feature = _test_offline.iloc[:,3:]
_test_offline_labels = _test_offline.iloc[:,1]
"""
_test_offline_feature, _test_offline_labels = split_train_label(_test_offline)

#get rid off del train data not because the data sort after PU wanna to save coding
#and release memory after

def positive_unlabel_learning(classifier, unlabel_data, threshold):
	score = classifier.predict_proba(unlabel_data.iloc[:,2:])
	score = pd.Series(score[:,1])
	score.loc[score >= threshold] = 1
	score.loc[score < threshold] = 0
	unlabel_data.insert(1, "label", score)
	print("\n# After PU found <{}> potential black instances".format(len(unlabel_data[unlabel_data.label == 1])))
	print("\n# After PU found <{}> potential white instances".format(len(unlabel_data[unlabel_data.label == 0])))
	return unlabel_data

def main():
	start = time.time()
	classifier = {
		"XGB" : XGBClassifier(max_depth = 4, n_estimators = 480, subsample = 0.8, gamma = 0.1,
							 scale_pos_weight =20, min_child_weight = 2,
							 colsample_bytree = 0.8, learning_rate = 0.08, n_jobs = -1),

	  	"logistic_regression" : LogisticRegression(penalty = "l2", C = 1, solver = "newton-cg",
	  						 class_weight = "balanced", max_iter = 300, n_jobs = -1),

		# NOTE:test min_samples_split and min_samples_leaf
		"random_forest" : RandomForestClassifier(n_estimators = 4, criterion = "entropy", max_depth = 13,
		 					 min_samples_split = 110, min_samples_leaf = 20, max_leaf_nodes = None,
							 verbose = 1, n_jobs = -1),

		"MLP" : MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
						     beta_1=0.9, beta_2=0.999, early_stopping=False,
						     epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
						     learning_rate_init=0.001, max_iter=200, momentum=0.9,
						     nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
						     solver='lbfgs', tol=0.0001, validation_fraction=0.1)
	}

	clf = classifier["random_forest"]

	with open(params_path  + "params.txt", 'a') as f:
		print("\n# Training clf :{}".format(clf))
		f.write(
		"**"*40 + "\n"*2
		+ str(clf) + "\n"*2
		+"**"*40 + "\n"*2
		)

	# NOTE: PU learning
	_clf = clf.fit(_train, _labels)
	#without PU offline score
	offline_score = offline_model_performance(_clf, _test_offline_feature, _test_offline_labels, params_path)
	unlabel_data = positive_unlabel_learning(_clf, _test_a, 0.8)
	#Choose Black Label
	unlabel_data = unlabel_data[unlabel_data.label == 0]
	#80% train data
	print(len(unlabel_data))
	print(len(_train_data))
	pu_train_data = file_merge(_train_data, unlabel_data, "date")
	print(len(pu_train_data))
	_new_train, _new_label = split_train_label(pu_train_data)

	#recall clf
	del clf
	clf = classifier["random_forest"]
	new_clf = clf.fit(_new_train, _new_label)
	del _new_train, _new_label
	probs = new_clf.predict_proba(_test_online)
	joblib.dump(clf, model_path + "{}.pkl".format("model"))
	#with PU offline score
	offline_score = offline_model_performance(new_clf, _test_offline_feature, _test_offline_labels, params_path)
	save_score(probs[:,1], score_path)

	# NOTE: Original
	"""
	_clf = clf.fit(_train, _labels)
	probs = clf.predict_proba(_test_online)
	joblib.dump(clf, model_path + "{}.pkl".format("model"))
	offline_score = offline_model_performance(clf, _test_offline_feature, _test_offline_labels, params_path)
	save_score(probs[:,1], score_path)
	"""
	# NOTE:  Feed Val Back
	"""
	# TODO: add xx in the future
	_train_data = pd.read_csv(train_path)
	_test_online = pd.read_csv(test_path)
	_train_data, _test_online = custom_imputation(_train_data, _test_online, fillna_value)
	#change -1 label to 1
	_train_data.loc[_train_data["label"] == -1] = 1
	"""

	print("\n# >>>>Duration<<<< : {}min ".format(round((time.time()-start)/60,2)))

if __name__ == '__main__':
	main()
