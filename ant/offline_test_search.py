import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
# from Xgboost

def offline_model_performance(estimator, validation_feature, validation_label):
    # if type(estimator) == string:
        # Load estimator
        # estimator =
    # Obtain array of false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(validation_label, estimator.predict(validation_feature)[:,1])
    # Search for tpr = 0.001
    fpr1 = 99
    fpr2 = 99
    fpr3 = 99
    for i in range(len(fpr)):
        if fpr[i] >= 0.001 and fpr[i] <= fpr1:
            fpr1 = fpr[i]
            tpr1 = tpr[i]
        elif fpr[i] >= 0.005 and fpr[i] <= fpr2:
            fpr2 = fpr[i]
            tpr2 = tpr[i]
        elif fpr[i] == 0.01 and fpr[i] <= fpr3:
            fpr3 = fpr[i]
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
    split_data = data[(data["date"] >= start_y_m_d) & (data["date"] <= end_y_m_d)]
    data = data.drop(data.index[(data["date"] >= start_y_m_d) & (data["date"] <= end_y_m_d)])
    return data, split_data

def main():
    data = pd.read_csv("data/train.csv")
    data.loc[data["label"] == -1] = 1
    print(data)
    print(data.loc[data["label"] == -1])
    test,test2 = test_train_split_by_date(data, 20170910, 20170911)
    print(test)
    print("training data :{}".format(len(test)/len(data.iloc[:,1]*100)))
    print("test data percentage :{}".format(len(test2)/len(data.iloc[:,1]*100)))

main()


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
