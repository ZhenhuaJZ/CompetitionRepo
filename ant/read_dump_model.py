import pandas as pd
from sklearn.externals import joblib

score_path = "log/last_3_days/{}d_{}h_{}m/".format(now.day, now.hour, now.minute)
params_path = "log/last_3_days/log_{}h.csv".format(now.hour)

train_path = "data/_train_data.csv"
test_b_path = "data/test_b.csv"
test_a_path = "data/test_a.csv"
validation_path = "data/_test_offline.csv"

model_path = ""

def load_dump_model(model_path):

    start = time.time()
    print("\n# START PU - TESTA , PU_thresh_a = {}".format(pu_thresh_a))
    test_a = pd.read_csv(test_a_path)
    clf = joblib.load(model_path)
    pu_black = positive_unlabel_learning(clf, test_a, pu_thresh_a)
    _train = file_merge(train, pu_black, "date")
    _feature, _label = split_train_label(_train)

    if eval:
        print("\n# EVAL PU")
        validation_path = "data/_test_offline.csv"
        validation = pd.read_csv(validation_path)
        val_feature, val_label = split_train_label(validation)
        val_probs = clf.predict_proba(val_feature)
        roc = offline_model_performance(val_label, val_probs[:,1])
        clear_mermory(val_feature, val_label, validation, validation_path, val_probs)

    if save_score:
        test_b = pd.read_csv(test_b_path)
        probs = clf.predict_proba(test_b.iloc[:,2:])
        score = pd.DataFrame(test_b["id"]).assign(score = probs[:,1])
        _score_path = score_path  + "pu_score_{}d_{}h_{}m.csv".format(now.day, now.hour, now.minute)
        score.to_csv(_score_path, index = None, float_format = "%.9f")
        print("\n# Score saved in {}".format(_score_path))
