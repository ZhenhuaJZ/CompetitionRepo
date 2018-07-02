import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics.scorer import make_scorer
import datetime
import numpy as np
import bisect
import re, csv

now = datetime.datetime.now()

############################### Offline performance ######################
def offline_model_performance(ground_truth, predict, **kwargs):
    # Obtain array of false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(ground_truth, predict)
    # NOTE: upper line
    fpr = fpr[2::2] #[1::2] #odd line (lower line)
    tpr = tpr[2::2]
    # Search for tpr at fpr = 0.001,0.005,0.01
    fpr1 = 99
    fpr2 = 99
    fpr3 = 99

    for i in range(0,len(fpr),1):
        min_dist_val = 0
        if fpr[i] >= 0.001 and fpr[i] <= fpr1:
            fpr_1 = fpr[i]
            tpr_1 = tpr[i]
            diff_n_1 = abs(fpr[i] - fpr[i-1])
            diff_p_1 = abs(fpr[i] - fpr[i+1])
            if (diff_p_1 < diff_n_1) and (diff_p_1 != 0):
                fpr_2 = fpr[i+1]
                tpr_2 = tpr[i+1]
            elif (diff_n_1 < diff_p_1) and (diff_n_1 != 0):
                fpr_2 = fpr[i-1]
                tpr_2 = tpr[i-1]
            elif diff_p_1:
                fpr_2 = fpr[i+1]
                tpr_2 = tpr[i+1]
            elif diff_n_1:
                fpr_2 = fpr[i-1]
                tpr_2 = tpr[i-1]
            else:
                continue
            fpr1 = fpr_1
            #print("tpr at 0.001 selected tpr_1:{:8f}, fpr_1:{:8f}, fpr_2:{:8f}, tpr_2:{:8f}".format(tpr_1, fpr_1, fpr_2, tpr_2))
            tpr1 = (((tpr_2-tpr_1)/(fpr_2-fpr_1))*(float(0.001)-fpr_1))+tpr_1
        elif fpr[i] >= 0.005 and fpr[i] <= fpr2:
            fpr_1 = fpr[i]
            tpr_1 = tpr[i]
            diff_n_1 = abs(fpr[i] - fpr[i-1])
            diff_p_1 = abs(fpr[i] - fpr[i+1])
            if (diff_p_1 < diff_n_1) and (diff_p_1 != 0):
                fpr_2 = fpr[i+1]
                tpr_2 = tpr[i+1]
            elif (diff_n_1 < diff_p_1) and (diff_n_1 != 0):
                fpr_2 = fpr[i-1]
                tpr_2 = tpr[i-1]
            elif diff_p_1:
                fpr_2 = fpr[i+1]
                tpr_2 = tpr[i+1]
            elif diff_n_1:
                fpr_2 = fpr[i-1]
                tpr_2 = tpr[i-1]
            else:
                continue
            fpr2 = fpr_1
            #print("tpr at 0.005 selected tpr_1:{:8f}, fpr_1:{:8f}, fpr_2:{:8f}, tpr_2:{:8f}".format(tpr_1, fpr_1, fpr_2, tpr_2))
            tpr2 = (((tpr_2-tpr_1)/(fpr_2-fpr_1))*(float(0.005)-fpr_1))+tpr_1
        elif fpr[i] >= 0.01 and fpr[i] <= fpr3:
            fpr_1 = fpr[i]
            tpr_1 = tpr[i]
            diff_n_1 = abs(fpr[i] - fpr[i-1])
            diff_p_1 = abs(fpr[i] - fpr[i+1])
            if (diff_p_1 < diff_n_1) and (diff_p_1 != 0):
                fpr_2 = fpr[i+1]
                tpr_2 = tpr[i+1]
            elif (diff_n_1 < diff_p_1) and (diff_n_1 != 0):
                fpr_2 = fpr[i-1]
                tpr_2 = tpr[i-1]
            elif diff_p_1:
                fpr_2 = fpr[i+1]
                tpr_2 = tpr[i+1]
            elif diff_n_1:
                fpr_2 = fpr[i-1]
                tpr_2 = tpr[i-1]
            else:
                continue
            fpr3 = fpr_1
            #print("tpr at 0.01 selected tpr_1:{:8f}, fpr_1:{:8f}, fpr_2:{:8f}, tpr_2:{:8f}".format(tpr_1, fpr_1, fpr_2, tpr_2))
            tpr3 = (((tpr_2-tpr_1)/(fpr_2-fpr_1))*(float(0.01)-fpr_1))+tpr_1

    model_performance = 0.4 * tpr1 + 0.3 * tpr2 + 0.3 * tpr3

    print("\n# Offline model performance_1 ROC : <<<{:9f}>>>".format(model_performance) + "\n"
          +"# fpr1 : {} ----> to tpr1: {:9f}".format(0.001, tpr1) + "\n"
          +"# fpr2 : {} ----> to tpr2: {:9f}".format(0.005, tpr2) + "\n"
          +"# fpr3 : {} ----> to tpr3: {:9f}".format(0.01, tpr3) + "\n"
    )


    with open(kwargs['params_path']  + "params.txt", 'a') as f:
        if len(kwargs) > 1 and isinstance(kwargs['fold'], int):
            f.write("Fold : <<<<" + str(kwargs['fold']) + ">>>>" +"\n")
        f.write(
        "**"*40 + "\n"*2
        +"Offline model performance_1 ROC(JL) : <<<{}>>>".format(str(model_performance)) + "\n"
        +"fpr1 : {} ----> to tpr1: {}".format(str(0.001), str(tpr1)) + "\n"
        +"fpr2 : {} ----> to tpr2: {}".format(str(0.005), str(tpr2)) + "\n"
        +"fpr3 : {} ----> to tpr3: {}".format(str(0.01), str(tpr3)) + "\n"
        +"##"*40 + "\n"*2
        )

    return model_performance

def get_tpr_from_fpr(fpr_array, tpr_array, target):
    fpr_index = np.where(fpr_array == target)
    assert target <= 0.01, 'the value of fpr in the custom metric function need lt 0.01'
    if len(fpr_index[0]) > 0:
        return np.mean(tpr_array[fpr_index])
    else:
        tmp_index = bisect.bisect(fpr_array, target)
        fpr_tmp_1 = fpr_array[tmp_index-1]
        fpr_tmp_2 = fpr_array[tmp_index]
        if (target - fpr_tmp_1) > (fpr_tmp_2 - target):
            tpr_index = tmp_index
        else:
            tpr_index = tmp_index - 1
        return tpr_array[tpr_index]

def get_tpr_from_fpr(fpr_array, tpr_array, target):
    fpr_index = np.where(fpr_array == target)
    assert target <= 0.01, 'the value of fpr in the custom metric function need lt 0.01'
    if len(fpr_index[0]) > 0:
        return np.mean(tpr_array[fpr_index])
    else:
        tmp_index = bisect.bisect(fpr_array, target)
        fpr_tmp_1 = fpr_array[tmp_index-1]
        fpr_tmp_2 = fpr_array[tmp_index]
        if (target - fpr_tmp_1) > (fpr_tmp_2 - target):
            tpr_index = tmp_index
        else:
            tpr_index = tmp_index - 1
        return tpr_array[tpr_index]

def offline_model_performance_2(ground_truth, predict, **kwargs):
    fpr, tpr, _ = roc_curve(ground_truth, predict, pos_label=1)
    tpr1 = get_tpr_from_fpr(fpr, tpr, 0.001)
    tpr2 = get_tpr_from_fpr(fpr, tpr, 0.005)
    tpr3 = get_tpr_from_fpr(fpr, tpr, 0.01)
    model_performance = 0.4*tpr1 + 0.3*tpr2 + 0.3*tpr3

    return model_performance

# #############################Log all the data ################################

def log_parmas(clf, valset, roc_1, roc_2, filename, fillna,  pu_thres = "n/a",
                cv_roc_1_mean = "n/a", cv_roc_2_mean = "n/a", under_samp = False, Feed_val_back = "True", mode = "PU"):
    #formate log
    valset = str(valset[0]) + "-" + str(valset[1])
    roc_1 = round(roc_1, 6)
    roc_2 = round(roc_2, 6)
    if isinstance(cv_roc_1_mean, float):
        cv_roc_1_mean = round(cv_roc_1_mean, 6)
        cv_roc_2_mean = round(cv_roc_2_mean, 6)
    filename = re.split('log/', filename)[-1]
    split_string = re.split('[(,' '\n)' ']', str(clf))

    f = csv.writer(open("log/log.csv", "a"))

    log = [] #clf name
    header = [] #header
    parmas = [] #parmas

    for i in split_string:
        if len(i)>0:
            new = i.strip().split('=')
            if len(new) <= 1:
                log.append(new[0])
            if len(new) > 1:
                header.append(new[0])
                parmas.append(new[1])

    #add header
    header.append("Validation set")
    header.append("ROC_1")
    header.append("ROC_2")
    header.append("5folds_ROC_1")
    header.append("5folds_ROC_2")
    header.append("Score")
    header.append("mode")
    header.append("file name")
    header.append("PU(threshold)")
    header.append("fillna")
    header.append("under_sampling")
    header.append("Feed val black back")

    #add content
    parmas.append(valset)
    parmas.append(roc_1)
    parmas.append(roc_2)
    parmas.append(cv_roc_1_mean)
    parmas.append(cv_roc_2_mean)
    parmas.append('')
    parmas.append(mode)
    parmas.append(filename)
    parmas.append(pu_thres)
    parmas.append(fillna)
    parmas.append(under_samp)
    parmas.append(Feed_val_back)

    f.writerow(log)
    f.writerow(header)
    f.writerow(parmas)
    f.writerow([])
