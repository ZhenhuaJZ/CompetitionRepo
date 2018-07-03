import pandas as pd
from lib.data_processing import *
from lib.model_performance import *
import datetime, time
import sys
now = datetime.datetime.now()
from xgboost import XGBClassifier


def partical_fit(data, feed_ratio, sort_by = ""):
	print("\n# Total length :", len(data))
	if sort_by != "":
		data = data.sort_values(by = str(sort_by))
		print("\n# Sort data in <{}> order".format(sort_by))
	partical_loc = int(len(data) * feed_ratio)
	data_seg_1 = data[:partical_loc]
	data_seg_2 = data[partical_loc:]
	print("\n# length of data_seg_1 :", len(data_seg_1))
	print("# length of data_seg_2 :", len(data_seg_2))
	clear_mermory(data)
	return data_seg_1, data_seg_2


classifier = {
"XGB" : XGBClassifier(max_depth = 4, n_estimators = 480, subsample = 0.8, gamma = 0,
min_child_weight = 1, scale_pos_weight = 1,
colsample_bytree = 0.8, learning_rate = 0.07, n_jobs = -1),
}
clf = classifier["XGB"]
log_path = "log/date_4/2:25_SM/"
score_path = log_path + "score/"
score_seg_1_path = score_path + "score_seg_a.csv"
test_path = "data/test_b.csv"


print("\n# Partical fit <test_b> to the dataset")
_test_online = df_read_and_fillna(test_path, 0)
test_b_seg_1,  test_b_seg_2 = partical_fit(_test_online, 0.5, "date")
clear_mermory(_test_online, test_b_seg_1)

#Read increment_train from hard drive
print("\n# Inititalize increment_train (read from hard drive)")
_train_data = pd.read_csv(increment_train_path, low_memory = False)
increment_train = df_read_and_fillna(increment_train_path)

#########################Merge Test_b score#################################
increment_train_feature, increment_train_label = split_train_label(increment_train)
clear_mermory(increment_train)
#Fit new classifier
clf.fit(increment_train_feature, increment_train_label)
clear_mermory(increment_train_feature, increment_train_label)

#Predict and save seg_2 score
prob_seg_2 = clf.predict_proba(test_b_seg_2.iloc[:,2:])
score_seg_2 = pd.DataFrame(test_b_seg_2["id"]).assign(score = prob_seg_2[:,1])

##############################Merge Score###################################
score_seg_1 = pd.read_csv(score_seg_1_path)
score = score_seg_1.append(score_seg_2)
score.to_csv(score_path + "score_day4_time2:25.csv", index = None, float_format = "%.9f")
print("\n# Score saved in {}".format(score_path))
