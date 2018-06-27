import numpy as np
import pandas as pd
import math
feature_starting_column = 2
save_link = '../../data/replaced_missing_train_complete.csv'

def round_to_whole(data, tolerance):
    p = 10**tolerance
    return int(data*p + 0.5)/p

def replace_missing_by_gaussian(data_array):
    # each column
    print("initiate printing")
    for i in range(data_array.iloc[0,:].size):
        # each feature
        if i < 298:
            continue
        count = 0
        mean = 0
        number_of_values = 0
        # Find mean
        for j in range(data_array.iloc[:,0].size):
            if pd.notna(data_array.iloc[j,i]):
                mean += data_array.iloc[j,i]
                number_of_values += 1
            count += 1
            if count % 5000 == 0:
                print("feature {}".format(i-1) + " processed mean {}".format(count) + " data")
        mean = mean/number_of_values
        print("mean is {}".format(mean))
        # Find the standard deviation
        count = 0
        standard_deviation = 0
        number_of_values = 0
        for j in range(data_array.iloc[:,0].size):
            if pd.notna(data_array.iloc[j,i]):
                standard_deviation += math.pow((data_array.iloc[j,i] - mean),2)
                number_of_values += 1
            count += 1
            if count % 5000 == 0:
                print("feature {}".format(i-1) + " processed standard deviation {}".format(count) + " data")
        standard_deviation = math.sqrt(standard_deviation/number_of_values)
        print("mean: {}".format(mean)+" standard deviation: {}".format(standard_deviation))

        count = 0
        NaN_number = 0
        for j in range(data_array.iloc[:,0].size):
            if pd.isna(data_array.iloc[j,i]):
                # rounding process and random normal are slowing the process down
                number = round_to_whole(np.random.normal(mean,standard_deviation),0)
                if number < 0:
                    data_array.iloc[j,i] = 0
                else:
                    data_array.iloc[j,i] = number
                    NaN_number += 1
            count += 1
            if count % 5000 == 0:
                print("feature {}".format(i-1) + " processed NaN {}".format(count) + " data")
    return data_array

def file_merge(black_data, white_data):
    merged_file = pd.concat([black_data,white_data], axis = 0)
    merged_file = merged_file.sort_values(by = "date")
    return merged_file


############################## Replace_missing by mode ############################
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
            #print(freq_diff)
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
        #if black_data["f{}".format(i)].mode()[0] == white_data["f{}".format(i)].mode()[0]:
            #common_mode = black_data["f{}".format(i)].mode()[0]
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
    print(white_data)
    return train_data_merged, test_data

# def main():
#     black_data_filled = pd.read_csv("../../data/black_label_mode_fill.csv")
#     print("read black data")
#     white_data_filled = pd.read_csv("../../data/white_label_mode_fill.csv")
#     print("read white data")
#     merged_data = file_merge(black_data_filled, white_data_filled)
#     merged_data.to_csv("../../data/train_mode_fill.csv",index = None)
# if __name__ == "__main__":
#     main()
