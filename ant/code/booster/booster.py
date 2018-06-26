"""
@Authors Leo.cui
7/5/2018
Xgboost

"""
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
import operator
import warnings

os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

#for i in range (65, 100, 4):
#*******************************hParams ***************************************#
now = datetime.datetime.now()

feats_selet_param = {

        "objective" : "binary:logistic",
        "max_depth" : 3,
        "min_child_weight" : 1,
        "gamma" : 0.1,
        "subsample" : 0.7,
        "colsample_bytree" : 1,
        #"alpha" : 0.05,
        "eta" : 0.08, #learning_rate
        #"eval_metric" : ['error','auc'], #early stop only effects on error (not sure which one its relay on)
        "num_round" : 400,
        "scale_pos_weight" : 10, #[60-70]
        "train_path" : "../../data/train_heatmap.csv", #[train_heatmap , train_mode_fill, train]
        "test_path" : "../../data/test_a_heatmap.csv", #[test_a_heatmap, test_a_mode_fill, test_a]
        "features_selection" : False,
        "importance_feats_rate" : 0.2, #if 0.8 then leave features importance > 80%
        }

param = {

        "objective" : "binary:logistic",
        "max_depth" : 4,
        "min_child_weight" : 2,
        "gamma" : 0.1,
        "subsample" : 0.8,
        "colsample_bytree" : 0.9,
        #"alpha" : 0.05,
        "eta" : 0.07, #learning_rate
        "eval_metric" : ['error','auc'], #early stop only effects on error (not sure which one its relay on)
        "num_round" : 480,
        "scale_pos_weight" : 10, #[60-70]
        "train_path" : "../../data/train.csv", #train_heatmap , train_mode_fill, train,
        "test_path" : "../../data/test_a.csv", #test_a_heatmap, test_a_mode_fill, test_a,
        #"features_selection" : True,
        #"importance_feats_rate" : 0.8, #if 0.8 then leave features importance > 80%
        }

num_round = param["num_round"]
features_selection = feats_selet_param["features_selection"]
importance_feats_rate = feats_selet_param["importance_feats_rate"]
early_stopping_rounds = 100
validation_mode = False # default 0.3's all data
suffix = "{}:{}".format(now.hour, now.minute) #signle training save file name's suffix

#*******************************************************************************#
#********************************Loop param*************************************#
#**********************if dont use loop set to False****************************#
#*******************************************************************************#


loop_function = False #if False shut down loop_function
loop_param = "eta" #change the loop parameter here
loop_start = 0.02 #start loop digit
loop_end = 0.1  #end loop digit
loop_step = 0.01  #loop stop


#***********************data_path***********************************************#
#train_data path
train_path = param["train_path"]
#test_data path
test_path = param["test_path"]


#***********************model & score path ***********************************************#
#model save path#
model_path = "model/"
#where to save the figure & answer & hParams
score_path = "score/{}_{}_{}:{}/".format(now.month, now.day, now.hour, now.minute)

#***********************Tool-box***********************************************#
#pandas read answer_sheet
#the path of answer_sheet.csv
as_path = "../../tool/answer_sheet.csv"
#xgb.fmap path
fmap = score_path+ "tool/"


def ceate_feature_map(df):
    print("\nCreating fmap")
    os.makedirs(fmap)
    features = df.columns.values
    outfile = open(fmap + "xgb.fmap", 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


def select_features(_train,_label,_test):
    info_file = score_path + "importance_features/"
    os.makedirs(info_file)
    print("\nStart selecting importance features")
    dtrain = xgb.DMatrix(_train, label=_label, nthread = -1)
    dtest = xgb.DMatrix(_test, nthread = -1)
    #dval = xgb.DMatrix(validation, label=validation_label, nthread = -1)
    bst = xgb.train(feats_selet_param, dtrain, num_round)
    
    importance = bst.get_fscore(fmap = fmap + "xgb.fmap")
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    feats_seleted  = df['feature'].tolist()
    f_score = df['fscore'].tolist()
    f_score.reverse()
    feats_seleted.reverse()
    slice_imp_feats = int(len(df['fscore'])*importance_feats_rate)
    _feats_seleted = feats_seleted[:slice_imp_feats] ## how many features should be left
    _f_score = f_score[:slice_imp_feats]
    #print(_f_score)
    #print(_feats_seleted)
    #After feature selection
    features_new = _train.loc[:,_feats_seleted]
    test_feature_new = _test.loc[:,_feats_seleted]


    with open(info_file + "importance_features.txt" , "w") as log:
        for f in range(len(feats_seleted)):
            log.write(str(f + 1) + "." +  " feature " +  str(feats_seleted[f]) + "  " + str(round(f_score[f] , 3)) + "\n")
        log.write(">"*30 + "Selected Features" + "<"*30 + "\n")

        for f in range(len(_feats_seleted)):
            log.write(str(f + 1) + "." +  " feature " +  str(_feats_seleted[f]) + "  " + str(round(_f_score[f] , 3)) + "\n")

    plt.figure(1, figsize=(70,25))
    plt.title("Feature Importance", fontsize = "40")
    plt.bar(range(len(feats_seleted)), f_score, color = "g", align = "center")
    plt.xticks(range(len(feats_seleted)), feats_seleted)
    plt.tick_params(axis='both', labelsize = '30')
    xlim = ([-1, len(feats_seleted)+20])
    plt.savefig(info_file + "f_score.png")
    print("\nFeatures selection done")

    plt.figure(2, figsize=(70,25))
    plt.title("Selected Feature Importance", fontsize = "40")
    plt.bar(range(len(_feats_seleted)), _f_score, color = "g", align = "center")
    plt.xticks(range(len(_feats_seleted)), _feats_seleted)
    plt.tick_params(axis='both', labelsize = '30')
    xlim = ([-1, len(_feats_seleted)+20])
    plt.savefig(info_file + "selected_f_score.png")
    """
    print("\nStart selecting importance features")
    xgb = XGBClassifier(n_estimators=2, max_depth=3, learning_rate = 0.07, 
                        subsample = 0.8, colsample_bytree = 0.9, n_jobs =-1)
    xgb = xgb.fit(features, labels)
    importances = xgb.feature_importances_
    print(importances)
    indices = np.argsort(importances)[::-1]

    model = SelectFromModel(xgb, prefit=True)
    features_new = model.transform(features)
    test_feature_new = model.transform(test_feature)
    with open(score_path + "importance_features.txt" , "w") as log:
        for f in range(features_new.shape[1]):
            log.write(str(f + 1) + "." +  " feature " +  str(indices[f]) + "  " + str(importances[indices[f]]) + "\n")
            #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print("Features selection done saved new data in data path")
    plt.figure(1, figure=(25,25))
    plt.title("Feature Importance")
    plt.bar(range(features_new.shape[1]), importances[indices], color = "g", align = "center")
    plt.xticks(range(features_new.shape[1]), features_new.columns[indices], rotation = 90)
    xlim = ([-1, features_new.shape[1]])
    plt.savefig(score_path + "f_score.png")
    """
    return features_new, test_feature_new

def plot_roc_curve(fpr, tpr ,label =None):
    linewidth = 8
    plt.plot(fpr, tpr, linewidth = linewidth, label = label)
    plt.plot([0,1],[0,1], 'k--')
    plt.plot([0.001,0.001],[1,0], 'r--', linewidth = linewidth)
    plt.plot([0.005,0.005],[1,0], 'r--', linewidth = linewidth)
    plt.plot([0.01,0.01],[1,0], 'r--', linewidth = linewidth)
    plt.plot([0,0.01],[0.35,0.35], 'g--', linewidth = linewidth)
    plt.plot([0,0.01],[0.45,0.45], 'g--', linewidth = linewidth)
    plt.yticks([0,0.1,0.2,0.3,0.32, 0.34, 0.35, 0.36, 0.37, 0.4,0.45,0.5,0.6,0.7,0.8,0.9])
    plt.xlim([0,0.01])
    plt.ylim([0,0.8])
    plt.tick_params(axis='both', labelsize = '35')
    plt.xlabel('FPR', fontsize = "35")
    plt.ylabel('TPR', fontsize = "35")
    plt.legend()

def _suffix():

    dict= {"min_child_weight" : "mcw",  "subsample" : "sb", 
           "colsample_bytree" : "cs", "max_depth" : "md", "eta" : "lr", "gamma" : "gamma", "scale_pos_weight" : "spw"}
    save_file_prefix = dict[loop_param]  # md = max_depth; sb = subsample; cs = colsample_bytree; mcw = min_child_weight

    return save_file_prefix

def parameters(loop_param_value):

    param.update({loop_param : loop_param_value})

    return param

def save_name_and_loop_param(loop_param_value):

    _save_file_prefix = _suffix()
    _param_name = str(_save_file_prefix) + "_" + str(loop_param_value)
    _param = parameters(loop_param_value)

    return _param, _param_name

def save_hParams (param, param_2, param_name):

    f = csv.writer(open(score_path + "hParams_{}.csv".format(param_name), "w"))
    for key, val in param.items():
        f.writerow([key, val])
    if features_selection :
        f2 = csv.writer(open(score_path + "feats_selet_hParams_{}.csv".format(param_name), "w"))
        for key, val in param_2.items():  
            f2.writerow([key, val])

    return print("\nhParams saved in : %s "  %(score_path))

def load_data():
    #Load train & test set

    train_data = pd.read_csv(train_path)
    train_data = train_data[(train_data.label==0)|(train_data.label==1)]
    test = pd.read_csv(test_path)
    #Define training set
    train = train_data.iloc[:,3:].fillna(1)
    label = train_data.iloc[:,1]
    test = test.iloc[:,2:].fillna(1)
    
    #scale_pos_weights = int(label.value_counts().values[0]/label.value_counts().values[1]) #scale_pos_weight
    
    ceate_feature_map(train)    
    
    if features_selection:
        train, test = select_features(train,label,test)

    if validation_mode :

        train, validation, label, validation_label = train_test_split(train, label, test_size = 0.3, random_state = 42)

        return train.values, test.values, label.values, validation.values, validation_label.values

    else:

        return train, test, label

def create_DMatrix(param, train, test, label, validation = None, validation_label = None):
#def create_DMatrix(param, train, test, label):

    dtrain = xgb.DMatrix(train, label=label, nthread = -1)

    dtest = xgb.DMatrix(test, nthread = -1)

    if validation_mode:

        dval = xgb.DMatrix(validation, label=validation_label, nthread = -1)

        watchlist = [(dval,'val'), (dtrain,'train')]

        evals_result = {}

        print("start training with validation...\n")

        bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=evals_result, early_stopping_rounds =  early_stopping_rounds)

        return bst, dtest, evals_result

    else:

        print("start training...\n")

        bst = xgb.train(param, dtrain, num_round)

        return bst, dtest
#computitional cost too much
#eval = xgb.cv(param, dtrain, num_round, nfold=6, metrics={'error'}, seed=0,  callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])

def predict_and_save_model(bst, dtest, param_name):
    #save model
    bst.save_model(model_path + "xbg_{}.model".format(param_name))
    #save xgboost structure
    #bst.dump_model(score_path + "xbg_{}.txt".format(param_name))

    preds = bst.predict(dtest, ntree_limit= bst.best_iteration)
    print("\nthe minimal loss found in : %i booster \n" %(bst.best_iteration))

    #features = pd.Series(bst.get_fscore()).sort_values(ascending=False)
    #save features and prepare feed into NN
    #features.to_csv(score_path + "xgb_features_{}.csv".format(param_name))

    #importance = bst.get_fscore(fmap = fmap)
    #print(importance)
    #importance = sorted(importance.items(), key=operator.itemgetter(1))
    #df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    #df['fscore'] = df['fscore'] / df['fscore'].sum()
    #print(df['fscore'])
    #print(df['feature'].tolist().reverse())
    #df.loc[]
    #plt.figure(1, figure=(25,25))
    #plt.title("Feature Importance")
    #plt.bar(df['fscore'].values, importances[indices], color = "g", align = "center")
    #plt.xticks(range(features_new.shape[1]), features_new.columns[indices], rotation = 90)
    #xlim = ([-1, features_new.shape[1]])
    #plt.savefig(score_path + "f_score.png")
    
    #xgb.plot_importance(bst)
    #plt.savefig(score_path + "f_score_{}.png".format(param_name))
    print("saving model & features ......\n")

    return preds

"""
importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

"""
#*******************************Save AUC/Error/Logloss***************************************#
def save_figure(param_name, evals_result):

    results = evals_result
    epochs = len(evals_result['val']['error'])
    x_axis = range(0, epochs)

    """
    plt.figure()
    plt.plot(x_axis, results['train']['logloss'], label='Train')
    plt.plot(x_axis, results['val']['logloss'], label='Validation')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.savefig(score_path + "LogLoss_{}.png".format(param_name))
    #plt.show()

    """

    plt.figure()
    plt.plot(x_axis, results['train']['error'], label='Train')
    plt.plot(x_axis, results['val']['error'], label='Validation')
    plt.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.savefig(score_path + "Error_{}.png".format(param_name))
    #plt.show()

    plt.figure()
    plt.plot(x_axis, results['train']['auc'], label='Train')
    plt.plot(x_axis, results['val']['auc'], label='Validation')
    plt.legend()
    plt.ylabel('AUC')
    plt.title('XGBoost AUC')
    plt.savefig(score_path + "AUC_{}.png".format(param_name))

#*******************************Save sore*************************************************#
def save_score(preds, param_name):

    low_risk = 0
    medium_risk = 0
    high_risk = 0

    for p in preds:
        if p > 0.2 and p < 0.4:
            low_risk += 1

        elif p >= 0.4 and p < 0.6:
            medium_risk += 1

        elif p >= 0.6:
            high_risk += 1

    print("probability [0.2 ~ 0.4] rate : {:.3f}%\n".format(100*(low_risk/len(preds))),
          "probability [0.4 ~ 0.6] rate : {:.3f}%\n".format(100*(medium_risk/len(preds))),
          "probability [0.6 ~ 1.0] rate : {:.3f}%\n".format(100*(high_risk/len(preds))))

    answer_sheet = pd.read_csv(as_path)
    #Dataframe data
    answer_sheet = pd.DataFrame(answer_sheet)
    #Feed result in score column
    answer = answer_sheet.assign(score = preds)
    #Save to .csv
    answer.to_csv(score_path + "{}_score_{}d{}m{}h.csv".format(param_name, now.day, now.month, now.hour), index = None, float_format = "%.9f")

    return print("Score saved in {}".format(score_path))


def loop_function_run():

    range = np.arange(loop_start, loop_end, loop_step)

    if validation_mode :
        train, test, label, validation, validation_label = load_data()
    else:
        train, test, label = load_data()

    for loop_param_value in range:

        loop_param_value = round(loop_param_value, 2)
    #for subsample in range (5, 10, 1): #when using subsample plz * 0.1. etc: 0.1*subsample = (0.5 - 1)
        print("XGBoost starting with loop_function, loop param is : %s , staring from : %s , end with : %s, step is : %s , run : %s\n"
               %(loop_param, loop_start, loop_end, loop_step, loop_param_value))

        _param, _param_name = save_name_and_loop_param(loop_param_value)
        save_hParams(_param, feats_selet_param, _param_name)

        if validation_mode :

            #train, test, label, validation, validation_label = load_data()
            bst, dtest, evals_result = create_DMatrix(_param, train, test, label, validation, validation_label)
            save_figure(_param_name, evals_result)
            plt.figure(figsize=(50,50))
            fpr, tpr, thresholds = roc_curve(validation_label, bst.predict(xgb.DMatrix(validation), ntree_limit= bst.best_iteration))
            plot_roc_curve(fpr, tpr, "auc")
            plt.savefig(score_path + "auc_{}.png".format(_param_name))

        else :

            #train, test, label = load_data()
            bst, dtest = create_DMatrix(_param, train, test, label)

        preds = predict_and_save_model(bst, dtest, _param_name)
        save_score(preds, _param_name)

        print("*******************************************Done***************************************************\n")

    return


def main():

    os.makedirs(score_path)

    if loop_function == True:

        loop_function_run()

    else:

        print("\nXGBoost starting with lr : %s" %(param["eta"]))
        save_hParams(param, feats_selet_param, suffix)

        if validation_mode :

            train, test, label, validation, validation_label = load_data()
            bst, dtest, evals_result = create_DMatrix(param, train, test, label, validation, validation_label)
            save_figure(suffix, evals_result)

            plt.figure(figsize=(50,50))
            fpr, tpr, thresholds = roc_curve(validation_label, bst.predict(xgb.DMatrix(validation), ntree_limit= bst.best_iteration))
            plot_roc_curve(fpr, tpr, "auc")
            plt.savefig(score_path + "auc.png")

        else :

            train, test, label = load_data()

            bst, dtest = create_DMatrix(param, train, test, label)

        preds = predict_and_save_model(bst, dtest, suffix)




        save_score(preds, suffix)

if __name__ == '__main__':
    main()
