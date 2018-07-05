import pandas as pd
from sklearn.externals import joblib

score_path = "log/last_3_days/{}d_{}h_{}m/".format(now.day, now.hour, now.minute)
params_path = "log/last_3_days/log_{}h.csv".format(now.hour)

train_path = "data/_train_data.csv"
test_b_path = "data/test_b.csv"
test_a_path = "data/test_a.csv"
validation_path = "data/_test_offline.csv"
filename = ""
model_path = "log/last_3_days/" + filename + "/inti_model.pkl"

pu_thresh_a_range = [0.95, 0.92, 0.91, 0.9, 0.88, 0.87, 0.7, 0.6]

def load_model():

    for pu_thresh_a in pu_thresh_a_range:

        test_a = pd.read_csv(test_a_path)
        train = pd.read_csv(train_path)
        pu_black = positive_unlabel_learning(clf, test_a, pu_thresh_a)
        _train = file_merge(train, pu_black, "date")
        _feature, _label = split_train_label(_train)
        clf = joblib.load(model_path)
        clf.fit(_feature, _label)
        print("\n# EVAL PU")
        validation_path = "data/_test_offline.csv"
        validation = pd.read_csv(validation_path)
        val_feature, val_label = split_train_label(validation)
        val_probs = clf.predict_proba(val_feature)
        roc = offline_model_performance(val_label, val_probs[:,1])

def main():
    load_model()

if __name__ == '__main__':
    main()
