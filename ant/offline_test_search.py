import pandas as pd
from sklearn.model_selection import cross_val_score
# from Xgboost

def offline_model_performance(estimator, validation_feature, validation_label):
    # if type(estimator) == string:
        # Load estimator
        # estimator =
    # Obtain array of false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(validation_label, estimator.predict(validation_feature)[:,1])
    # Search for tpr = 0.001
    for i in len(fpr):
        if fpr[i] == 0.001:
            tpr1 = tpr[i]
        elif fpr[i] == 0.005:
            tpr2 = tpr[i]
        elif fpr[i] == 0.01:
            tpr3 = tpr[i]
    model_performance = 0.4 * tpr1 + 0.3 * tpr2 + 0.3 * tpr3
    return model_performance

def batch_data(data, split_ratio):
    size_per_batch = len(data.iloc[:,1]) * split_ratio
    num_batch = int(1/split_ratio)
    batch = {}
    for i in range(num_batch):
        batch["batch_{}".format(i)] = data.loc[(i*size_per_batch):(size_per_batch*(i+1))]
    return batch

def test_train_split_by_date(data, start_y_m_d, end_y_m_d):
    split_data = data[(data["date"] >= start_y_m_d) && (data["date"] <= end_y_m_d)]
    data = data.drop(data.index[(data["date"] == start_y_m_d):(data["date"] == end_y_m_d)])
    return data, split_data



# def test_set_search(estimator, data, online_score, test_set_size):
#     min_diff = 0
#     feature = data.iloc[:,3:]
#     label = data.iloc[:,1]
#     for batch in all_batch:
#         offline_score = offline_model_performance(estimator, all_batch[batch].iloc[:,3:], all_batch[batch].iloc[:,1])
#         diff = abs(online_score - offline_score)
#         if diff < min_diff:
#             min_diff = diff
#             offline_data = all_batch[batch]
#
#     return offline_data, similarity
