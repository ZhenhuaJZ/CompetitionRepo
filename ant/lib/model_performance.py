import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
import datetime
now = datetime.datetime.now()

def offline_model_performance(estimator, validation_feature, validation_label, params_path):
    #log_path = "log/date_{}/GS_{}:{}/".format(now.day,now.hour,now.minute)
    #params_path = log_path + "params/"
    # Obtain array of false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(validation_label, estimator.predict_proba(validation_feature)[:,1])
    fpr = fpr[1::2]
    tpr = tpr[1::2]
    #print("fpr list", fpr[:20])
    #print("tpr list", tpr[:20])
    #print("fpr list lens{}".format(len(fpr)))
    #print("tpr list lens{}".format(len(tpr)))
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
            print("tpr at 0.001 selected tpr_1:{:7f}, fpr_1:{:7f}, fpr_2:{:7f}, tpr_2:{:7f}".format(tpr_1, fpr_1, fpr_2, tpr_2))
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
            print("tpr at 0.005 selected tpr_1:{:7f}, fpr_1:{:7f}, fpr_2:{:7f}, tpr_2:{:7f}".format(tpr_1, fpr_1, fpr_2, tpr_2))
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
            print("tpr at 0.01 selected tpr_1:{:7f}, fpr_1:{:7f}, fpr_2:{:7f}, tpr_2:{:7f}".format(tpr_1, fpr_1, fpr_2, tpr_2))
            tpr3 = (((tpr_2-tpr_1)/(fpr_2-fpr_1))*(float(0.01)-fpr_1))+tpr_1

    """
    for i in range(1,len(fpr),2):
        min_dist_val = 0
        if fpr[i] >= 0.001 and fpr[i] <= fpr1:
            fpr_1 = fpr[i]
            tpr_1 = tpr[i]
            diff_n_1 = abs(fpr[i] - fpr[i-2])
            diff_p_1 = abs(fpr[i] - fpr[i+2])
            if (diff_p_1 < diff_n_1) and (diff_p_1 != 0):
                fpr_2 = fpr[i+2]
                tpr_2 = tpr[i+2]
            elif (diff_n_1 < diff_p_1) and (diff_n_1 != 0):
                fpr_2 = fpr[i-2]
                tpr_2 = tpr[i-2]
            elif diff_p_1:
                fpr_2 = fpr[i+2]
                tpr_2 = tpr[i+2]
            elif diff_n_1:
                fpr_2 = fpr[i-2]
                tpr_2 = tpr[i-2]
            else:
                continue
            fpr1 = fpr_1
            print("tpr at 0.001 selected tpr_1:{:7f}, fpr_1:{:7f}, fpr_2:{:7f}, tpr_2:{:7f}".format(tpr_1, fpr_1, fpr_2, tpr_2))
            tpr1 = (((tpr_2-tpr_1)/(fpr_2-fpr_1))*(float(0.001)-fpr_1))+tpr_1
        elif fpr[i] >= 0.005 and fpr[i] <= fpr2:
            fpr_1 = fpr[i]
            tpr_1 = tpr[i]
            diff_n_1 = abs(fpr[i] - fpr[i-2])
            diff_p_1 = abs(fpr[i] - fpr[i+2])
            if (diff_p_1 < diff_n_1) and (diff_p_1 != 0):
                fpr_2 = fpr[i+2]
                tpr_2 = tpr[i+2]
            elif (diff_n_1 < diff_p_1) and (diff_n_1 != 0):
                fpr_2 = fpr[i-2]
                tpr_2 = tpr[i-2]
            elif diff_p_1:
                fpr_2 = fpr[i+2]
                tpr_2 = tpr[i+2]
            elif diff_n_1:
                fpr_2 = fpr[i-2]
                tpr_2 = tpr[i-2]
            else:
                continue
            fpr2 = fpr_1
            print("tpr at 0.005 selected tpr_1:{:7f}, fpr_1:{:7f}, fpr_2:{:7f}, tpr_2:{:7f}".format(tpr_1, fpr_1, fpr_2, tpr_2))
            tpr2 = (((tpr_2-tpr_1)/(fpr_2-fpr_1))*(float(0.005)-fpr_1))+tpr_1
        elif fpr[i] >= 0.01 and fpr[i] <= fpr3:
            fpr_1 = fpr[i]
            tpr_1 = tpr[i]
            diff_n_1 = abs(fpr[i] - fpr[i-2])
            diff_p_1 = abs(fpr[i] - fpr[i+2])
            if (diff_p_1 < diff_n_1) and (diff_p_1 != 0):
                fpr_2 = fpr[i+2]
                tpr_2 = tpr[i+2]
            elif (diff_n_1 < diff_p_1) and (diff_n_1 != 0):
                fpr_2 = fpr[i-2]
                tpr_2 = tpr[i-2]
            elif diff_p_1:
                fpr_2 = fpr[i+2]
                tpr_2 = tpr[i+2]
            elif diff_n_1:
                fpr_2 = fpr[i-2]
                tpr_2 = tpr[i-2]
            else:
                continue
            fpr3 = fpr_1
            print("tpr at 0.01 selected tpr_1:{:7f}, fpr_1:{:7f}, fpr_2:{:7f}, tpr_2:{:7f}".format(tpr_1, fpr_1, fpr_2, tpr_2))
            tpr3 = (((tpr_2-tpr_1)/(fpr_2-fpr_1))*(float(0.01)-fpr_1))+tpr_1
        """
    model_performance = 0.4 * tpr1 + 0.3 * tpr2 + 0.3 * tpr3
    print("\n# Offline model performance ROC : {}".format(model_performance))
    print("\n# Perfromance ROC : <<<{}>>>".format(str(model_performance)) + "\n"
          +"# fpr1 : {} ----> to tpr1: {}".format(str(0.001), str(tpr1)) + "\n"
          +"# fpr2 : {} ----> to tpr2: {}".format(str(0.005), str(tpr2)) + "\n"
          +"# fpr3 : {} ----> to tpr3: {}".format(str(0.01), str(tpr3)) + "\n"
    )
    with open(params_path  + "params.txt", 'a') as f:
        f.write(
        "**"*40 + "\n"*2
        +"Perfromance ROC : <<<{}>>>".format(str(model_performance)) + "\n"
        +"fpr1 : {} ----> to tpr1: {}".format(str(0.001), str(tpr1)) + "\n"
        +"fpr2 : {} ----> to tpr2: {}".format(str(0.005), str(tpr2)) + "\n"
        +"fpr3 : {} ----> to tpr3: {}".format(str(0.01), str(tpr3)) + "\n"
        +"**"*40 + "\n"*2
        )
    return model_performance
