import os
import numpy as np
import pandas as pd
import math
import datetime
import gc
from imblearn.over_sampling import SMOTE
now = datetime.datetime.now()

##################### Data subsampling / imbalanced data ######################

#standarlization data
def standarlization(df):
    df = (df - df.mean()) / (df.max() - df.min())
    return df
# The ratio defines what is the output label 1 and 0 ratio
def under_sampling(data, ratio = 1):
    label_1_data = data.loc[data["label"] == 1]
    label_1_size = len(label_1_data["label"])
    label_0_data = data.drop(data.index[data["label"] == 1])
    label_0_data = label_0_data.sample(n = int(label_1_size*ratio), random_state = 10)
    under_sample_data = file_merge(label_1_data, label_0_data, sort_by = "date", reset_index = True)
    print("\n# Number of label 1 and 0:\n", under_sample_data["label"].value_counts())
    return under_sample_data

def populate_1 (data, num_label_1):
	new_label_1 = pd.DataFrame(1, index = pd.Series(range(0,num_label_1)), columns = data.columns.values)
	data = file_merge(data, new_label_1, sort_by = "date" , reset_index = True)
	return data

def over_sampling(data, ratio = 1):
    label_1_data = data.loc[data["label"] == 1]
    label_1_size = len(label_1_data["label"])
    sampled_data = label_1_data.sample(n = int(label_1_size*ratio), replace = True, random_state = 10)
    over_sampled_data = file_merge(data, sampled_data, sort_by = "date", reset_index = True)
    print("\n# Original data number of label 1:{}, label 0:{}".format(len(data.loc[data["label"] == 1]),len(data.loc[data["label"] == 0])))
    print("\n# After over sample number of label 1:{}, label 0:{}".format(
            len(over_sampled_data.loc[over_sampled_data["label"] == 1]),len(over_sampled_data.loc[over_sampled_data["label"] == 0])))
    return over_sampled_data

#TODO: SMOTE sampling technique
# Synthetic Minority Over-sampling technique
def SMOTE_sampling(data, ratio = 1):
    pd.options.mode.chained_assignment = None
    print("\n# Initiate SMOTE over sampling")
    sm = SMOTE(ratio = "minority", random_state = 5)
    id = data["id"]
    feature = data.drop(columns = ["label", "id"])
    header = feature.columns.values
    label = data.iloc[:,1]
    num_label_1 = len(label[label == 1])
    num_label_0 = len(label[label == 0])
    print("\n# Before sampling: label 1 = {}, label 0 = {}".format(num_label_1, num_label_0))
    new_feature, new_label = sm.fit_sample(feature,label)
    new_feature = pd.DataFrame(new_feature, columns = header)
    new_label = pd.Series(new_label)
    new_feature.insert(0, value = new_label, column = "label")
    new_feature.insert(0, value = id, column = "id")
    over_sampled_data = new_feature.drop(new_feature.index[-int(num_label_0*(1-ratio)):])
    over_sampled_data.iloc[:,1:] = over_sampled_data.iloc[:,1:].astype("int32")
    over_sampled_data.iloc[:,3:] = over_sampled_data.iloc[:,3:].astype("float32")
    over_sampled_data = over_sampled_data.sort_by("date").reset_index()
    sampled_num_label_1 = len(over_sampled_data.loc[over_sampled_data["label"] == 1])
    sampled_num_label_0 = len(over_sampled_data.loc[over_sampled_data["label"] == 0])
    print("\n# After SMOTE sampling: label 1 = {}, label 0 = {}".format(sampled_num_label_1, sampled_num_label_0))
    print("\n# Added total number of label 1 = {}".format(sampled_num_label_1-num_label_1))
    print("\n# End of SMOTE sampling")
    print(type(over_sampled_data['label']))
    pd.options.mode.chained_assignment = "warn"
    return over_sampled_data

# TODO: uncompleted
# def SMOTETomek(data):
#     sm = SMOTE(random_state = 2)
#     # feature = data.drop(columns = "label")
#     feature = data.iloc[:, 3:]
#     label = data.iloc[:,1]
#     new_feature, new_label = sm.fit_sample(feature,label)
#     print(new_feature)
#     print(new_label.value_counts())
#     data = feature.insert(1, "label", new_label)
#     print(data)
#     return data

########################### Memory Manage ######################################
def clear_mermory(*args):
    for a in args:
        del a
    gc.collect()

#check the dtype of a dataframe
def dataframe_management():
    #df.iloc[:,1:] = df.iloc[:,:].astype('int32')
    df.info(memory_usage='deep')
    for dtype in ['float','int','object']:
        selected_dtype = _train_data.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

# #####################Creat path###############################################
def creat_project_dirs(log_path):
    params_path = log_path + "params/"
    score_path = log_path + "score/"
    model_path = log_path + "model/"
    os.makedirs(log_path)
    os.makedirs(params_path)
    os.makedirs(score_path)
    os.makedirs(model_path)

def round_to_whole(data, tolerance):
    p = 10**tolerance
    return int(data*p + 0.5)/p

#################################### Split data ################################
def batch_data(data, split_ratio):
    size_per_batch = len(data.iloc[:,1]) * split_ratio
    num_batch = int(1/split_ratio)
    batch = {}
    for i in range(num_batch):
        batch["batch_{}".format(i)] = data.loc[(i*size_per_batch):(size_per_batch*(i+1))]
    return batch

def sample_segmentation(data, feature_list):
    # seg_a_data is data that is larger than value_range
    # seg_b_data is data that is less than value_range
    seg_a_data = data
    # Interate through each feature in the feature_list
    # If the each feature_list is larger than value_range
    # Then it is stored inside seg_a_data
    for feature in feature_list:
        seg_a_data = seg_a_data.loc[seg_a_data[feature] >= feature_list[feature]]
    # create seg_b_data
    seg_b_data = data.drop(seg_a_data.index)
    return seg_a_data, seg_b_data

#Pass the training dataframe or datapath and split to feature and label
def split_train_label(data):
    #print(type(data))
    if isinstance(data, str):
        data = pd.read_csv(data)
        feature = data.iloc[:,3:]
        label = data.iloc[:,1]
    elif isinstance(data, pd.core.frame.DataFrame):
        feature = data.iloc[:,3:]
        label = data.iloc[:,1]
    return feature, label

# test_train_split_by_date split the test set by providing a range of dates in yyyymmdd
def test_train_split_by_date(data, start_y_m_d, end_y_m_d):
    # Extract data sets within the start and end date
    split_data = data[(data["date"] >= start_y_m_d) & (data["date"] <= end_y_m_d)]
    # Remove the data sets within the starting and end date
    data = data.drop(data.index[(data["date"] >= start_y_m_d) & (data["date"] <= end_y_m_d)])
    split_data_percent = round(len(split_data)/len(data.iloc[:,1]),2) * 100
    # Reset both data set index to count from zero
    print("\n# Split by date from <<<{}>>> to <<<{}>>>".format(str(start_y_m_d), str(end_y_m_d)))
    print("\n# Offline test percentage {}%".format(split_data_percent))
    print("\n# Number of label 0 and 1 in test set:\n", split_data["label"].value_counts())
    print("\n# Percentage of label 0 and 1 in test set:\n", split_data["label"].value_counts()* 100/len(split_data["label"]))

    return data, split_data

# This function merges two dataframe and can be sort by provided string
def file_merge(data_1, data_2, sort_by = "", reset_index = False):
    if len(data_1) == 0:
        print("\n# Waring merge file 1 has 0 lengh !")
        return data_2
    elif len(data_2) == 0:
        print("\n# Waring merge file 2 has 0 lengh !")
        return data_1
    merged_file = pd.concat([data_1,data_2], axis = 0)
    clear_mermory(data_1, data_2)
    if sort_by != "":
        merged_file = merged_file.sort_values(by = str(sort_by))
        print("\n# Merged data in <{}> order".format(sort_by))
    if reset_index:
        merged_file.reset_index(inplace = True, drop = True)
        print("\n# Merged data and sort in <Index Order>")
    return merged_file

# Merge all the files under srcpath and save to despath
def file_merge_hard_drive(srcpath, despath):
    files = os.listdir(srcpath)
    print("\n# Merge {}",format(files))
    #files = ['merge_train.csv','labeled_with_time.csv']
    with open(despath, 'w+') as output:
        for eachfile in files:
            print(eachfile)
            filepath = os.path.join(srcpath, eachfile)
            print(filepath)
            with open(filepath, 'r+') as infile:
                data = infile.read()
                output.write(data)
    print("# File merge done")

def df_read_and_fillna(data_path, fillna_value = 0):
	data = pd.read_csv(data_path)
	data = custom_imputation(data, fillna_value)
	return data

############################## Replace_missing by mode #########################
# This function still having trouble
def find_common_mode(black_frequency_list, white_frequency_list):
    min_freq_diff = 99999;
    min_mode_value = 0;
    # Find the most similar common mode between black_frequency_list and white_frequency_list
    for i in range(5):
        if i == len(white_frequency_list)-1:
            break;
        for j in range(5):
            if j == len(black_frequency_list)-1:
                break;
            # Calculate the difference in frequency, the number with smallest frequency is stored in min_mode_value
            freq_diff = abs((white_frequency_list.iloc[i] - black_frequency_list.iloc[j])*100)
            if (freq_diff < min_freq_diff) and (white_frequency_list.index[i] == black_frequency_list.index[j]):
                min_mode_value = white_frequency_list.index[i]
    common_mode = min_mode_value
    return common_mode

def replace_missing_by_custom_mode(train_data,test_data):
    print("Initiate custom mode filling nan process")
    black_data = train_data.loc[train_data["label"] == 1]
    white_data = train_data.loc[train_data["label"] == 0]
    for i in range(black_data.shape[1]-3):
        col_name = black_data.columns.values.tolist()[3:]
        if black_data[col_name[i]].mode()[0] == white_data[col_name[i]].mode()[0]:
            common_mode = black_data[col_name[i]].mode()[0]
        else:
            # Calculate a list of occurrence in each feature
            black_mode_list = black_data[col_name[i]].value_counts() / len(black_data[col_name[i]])
            white_mode_list = white_data[col_name[i]].value_counts() / len(white_data[col_name[i]])
            # Calculate the common occurrence between black and white data
            common_mode = find_common_mode(black_mode_list,white_mode_list)
        black_data[col_name[i]] = black_data[col_name[i]].fillna(black_data[col_name[i]].mode()[0])
        white_data[col_name[i]] = white_data[col_name[i]].fillna(white_data[col_name[i]].mode()[0])
        test_data[col_name[i]] = test_data[col_name[i]].fillna(common_mode)
        print("Filled feature {}".format(col_name[i]) + "***Black Filled:{}".format(black_data[col_name[i]].mode()[0]) +
              "***White Filled:{}".format(white_data[col_name[i]].mode()[0]) + "***Test Filled:{}".format(common_mode))
        print("******************************")
    print("End of custom mode filling")
    train_data_merged = file_merge(black_data, white_data)
    return train_data_merged, test_data

#custom_imputation
def custom_imputation_3_inputs(df_train, df_test_b, df_test_a, fillna_value = 0):
    train = df_train.fillna(fillna_value)
    test_b = df_test_b.fillna(fillna_value)
    test_a = df_test_a.fillna(fillna_value)
    print("##"*40)
    print("\n# Filling missing data with <{}>".format(fillna_value))
    return train, test_b, test_a

#custom_imputation
def custom_imputation(df, fillna_value = 0):
    data = df.fillna(fillna_value)
    #print("##"*40)
    #print("\n# Filling missing data with <{}>".format(fillna_value))
    return data
# #############################Save score#######################################
#pass preds and save score file path
"""
def save_score(preds, score_path):
    as_path = "lib/answer_sheet.csv"
    answer_sheet = pd.read_csv(as_path)
    answer_sheet = pd.DataFrame(answer_sheet)
    answer = answer_sheet.assign(score = preds)
    answer.to_csv(score_path + "score_day{}_time{}:{}.csv".format(now.day, now.hour, now.minute), index = None, float_format = "%.9f")
    return print("\n# Score saved in {}".format(score_path))
"""
def save_score(clf, test_path, score_path, feature_drops, prefix):
    test_data = pd.read_csv(test_path)
    if len(feature_drops) != 0:
        test_data = test_data.drop(feature_drops, axis = 1)
    probs = clf.predict_proba(test_data.iloc[:,2:])
    score = pd.DataFrame(test_data["id"]).assign(score = probs[:,1])
    _score_path = score_path  + "{}_score_{}d_{}h_{}m.csv".format(prefix, now.day, now.hour, now.minute)
    score.to_csv(_score_path, index = None, float_format = "%.9f")
    return print("\n# Score saved in {}".format(_score_path))
