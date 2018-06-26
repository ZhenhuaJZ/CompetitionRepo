"""
@Authors Leo.cui
7/5/2018
Format train data

"""

import pandas as pd
import numpy as np
import os

def get_score_format(test_path, answer_sheet_path):
    #read_data
    df = pd.read_csv(test_path)

    answer_sheet = df['id']

    #save answer_sheet csv
    answer_sheet.to_csv(answer_sheet_path, index = None, header = True)

    return


def xbg_format(data_path, save_path, sort_data = True, fillzero = True, header = True, label_mode = None):

    #read_data
    df = pd.read_csv(data_path)

    #get ride off -1 label
    if label_mode == "keep_0_1":
        df = df[(df.label==0)|(df.label==1)]
    elif label_mode == "keep_1":
        df = df[(df.label==1)]
    elif label_mode == "keep_0":
        df = df[(df.label==0)]
    elif label_mode == "keep_-1":
        df = df[(df.label== -1)]
    

    #sorting by data
    if sort_data == True:
        df.sort_values('date', inplace = True)

    #slicing index first column of the data
    #df = df.iloc[:,0:3]

    #delete data column
    #delete_col = np.load("./corr_data/heat_map_del.npy")
    #df = df.drop(['date','id'], axis=1)
    #df = df.drop(delete_col.tolist(), axis=1)

    if fillzero == True:
        #fill na
        df = df.fillna(0)


    #save csv
    #print(df)
    df.to_csv(save_path, index = None, header = header)

    return


def csv2npy(csv_path, npy_path):

    _csv = np.loadtxt(csv_path, delimiter=',')

    #deal with missing data
    #_csv = np.genfromtxt(csv_path, delimiter=',', missing_values = "N/A", filling_values = np.nan)

    #_csv = np.genfromtxt(csv_path, delimiter=",", filling_values = -999)

    np.save(npy_path, _csv)


def mergeFile(srcpath, despath):
    '将src路径下的所有文件块合并，并存储到des路径下。'
    
    #files = os.listdir(srcpath)
    files = ['train_1_mode_fill.csv','train_0_mode_fill.csv']

    #file = []

    with open(despath, 'w+') as output:
        for eachfile in files:
            print(eachfile)
            filepath = os.path.join(srcpath, eachfile)
            print(filepath)
            with open(filepath, 'r+') as infile:
                data = infile.read()
                output.write(data)


def convert_to_date(array):
    list = []
    for i in range(len(array)):
        value = array[i] % 100
        list.append(value)
    return list

def main():

    #"/home/leo/ant_leo/data/replaced_missing_train_complete.csv"
    data_path = "train.csv"
    save_path = "train_fill_0.csv"

    csv_path = "test_a_heatmap.csv"
    npy_path = "test_a_heatmap.npy"

    #test_path = "/home/leo/ant/model/data/test_a.csv"
    #answer_sheet_path = "/home/leo/ant/score/answer_sheet.csv"

    srcpath = "merge/"
    despath = "train_mode_fill.csv"

    train_path = "train.csv"
    new_train_save_path = "train_added_f298_1.csv"
    test_path = ""
    new_test_save_path = ""

    xbg_format(data_path, save_path, 
               sort_data = True, fillzero = True, 
               header = True, label_mode = None) ## label_mode = [keep_0_1, keep_1, keep_0, keep_-1]
    #mergeFile(srcpath, despath)
    #csv2npy(csv_path, npy_path)
    #get_score_format(test_path, answer_sheet_path)

    #train_data = pd.read_csv(train_path)
    #train_data_new_feature = convert_to_date(train_data["date"])
    #train_data['f298'] = train_data_new_feature
    #train_data.to_csv(new_train_save_path, index = None)

    #test_data = pd.read_csv(test_path)
    #test_data_new_feature = convert_to_date(test_data["date"])
    #test_data['f298'] = test_data_new_feature
    #test_data.to_csv(new_test_save_path, index = None)

if __name__ == '__main__':
    main()
