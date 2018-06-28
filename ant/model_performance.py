import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve

def offline_model_performance(estimator, validation_feature, validation_label):
    now = datetime.datetime()
    log_path = "log/date_{}/GS_{}:{}/".format(now.day,now.hour,now.minute)
    params_path = log_path + "params/"
    # Obtain array of false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(validation_label, estimator.predict_proba(validation_feature)[:,1])
    # Search for tpr at fpr = 0.001,0.005,0.01
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
        elif fpr[i] >= 0.01 and fpr[i] <= fpr3:
            fpr3 = fpr[i]
            tpr3 = tpr[i]
    model_performance = 0.4 * tpr1 + 0.3 * tpr2 + 0.3 * tpr3
    print("\n# Offline model performance ROC : {}".format(model_performance))
    with open(params_path  + "params.txt", 'a') as f:
        f.write(
        "**"*40 + "\n"*2
        +"fpr1 : {} ----> to tpr1: {}".format(str(fpr1), str(tpr1)) + "\n"
        +"fpr2 : {} ----> to tpr2: {}".format(str(fpr2), str(tpr2)) + "\n"
        +"fpr3 : {} ----> to tpr3: {}".format(str(fpr3), str(tpr3)) + "\n"
        +"**"*40 + "\n"*2
        )
    return model_performance
