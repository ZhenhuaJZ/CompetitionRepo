import numpy as np
import pandas as pd
import math
feature_starting_column = 2
save_link = '../../data/replaced_missing_train_complete.csv'

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

# test_train_split_by_date split the test set by providing a range of dates in yyyymmdd
def test_train_split_by_date(data, start_y_m_d, end_y_m_d):
    split_data = data[(data["date"] >= start_y_m_d) & (data["date"] <= end_y_m_d)]
    data = data.drop(data.index[(data["date"] >= start_y_m_d) & (data["date"] <= end_y_m_d)])
    print("\n# Offline test percentiage {}%".format(len(split_data)/len(data.iloc[:,1]*100)))
    return data, split_data

# This function merges two dataframe and can be sort by provided string
def file_merge(data_1, data_2, sort_by = 0, reset_index = False):
    merged_file = pd.concat([data_1,data_2], axis = 0)
    if sort_by != 0:
        merged_file = merged_file.sort_values(by = sort_by)
    if reset_index:
        merged_file.reset_index()
    return merged_file

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
