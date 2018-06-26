import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import Imputer, StandardScaler, Normalizer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.metrics import roc_curve, roc_auc_score
import warnings

param = {
        "objective" : "binary:logistic",
        "max_depth" : 4,
        "subsample" : 0.9,
        "colsample_bytree" : 1,
        "min_child_weight" : 1,
        "gamma" : 0.1,
        "eta" : 0.06, #learning_rate
        "eval_metric" : ['error'], #early stop only effects on error
        "silent" : 0
        }

num_round = 480

#***********************model & score path *************************************#
#model save path#
model_path = "model/"
#where to save the figure & answer & hParams
score_path = "score/"

#***********************data_path**********************************************#
data_path = path1 + "/data/"

train_path = "../data/train_heatmap.csv"
#test_data path
test_path = "../data/test_heatmap.csv"

unlabel_path = "../data/unlabel.csv"

stack_test_path = score_path + "stack_test_sheet.csv"

now = datetime.datetime.now()
log_file = "log/"+ "{}_{}_{}/".format(now.year, now.month, now.day)
sub_log_file = log_file + "{}:{}:{}/".format(now.hour, now.minute, round(now.second,2))


#fmap = path1 + "/fmap/xgb.fmap"


def plot_roc_curve(fpr, tpr ,label =None):
	plt.plot(fpr, tpr, linewidth =2, label = label)
	plt.plot([0,1],[0,1], 'k--')
	plt.plot([0.001,0.001],[1,0], 'r--')
	plt.plot([0.005,0.005],[1,0], 'r--')
	plt.plot([0.01,0.01],[1,0], 'r--')
	plt.plot([0,0.01],[0.35,0.35], 'g--')
	plt.plot([0,0.01],[0.45,0.45], 'g--')
	plt.yticks([0,0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9])
	plt.xlim([0,0.01])
	plt.ylim([0,0.8])
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.legend()

def progress_log(names = None, classifiers = None, name = None, end_log = False, start_log = False):
    #now = datetime.datetime.now()
    #log_file = "log/"+ "{}_".format(now.year)+"{}_".format(now.month)+"{}/".format(now.day)
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    #else:
    if end_log:
        with open(log_file + "{}_".format(now.year)+"{}_".format(now.month)+"{}".format(now.day), "a") as f:
            f.write(str('>'*40)+  str("END LOG") + str('<'*40) + '\n')
            f.write(str('>'*40)+  str("{}:{}".format(now.hour, now.minute)) + str('<'*40) + '\n')
            f.write('\n')
            f.write('\n')
    elif start_log:
        os.makedirs(sub_log_file)
        with open(log_file + "{}_".format(now.year)+"{}_".format(now.month)+"{}".format(now.day), "a") as f:
            f.write(str('>'*40)+  str("Start LOG") + str('<'*40) + '\n')
            f.write(str('>'*40)+  str("{}:{}".format(now.hour, now.minute)) + str('<'*40) + '\n')
            f.write('\n')
    else:
        with open(log_file + "{}_".format(now.year)+"{}_".format(now.month)+"{}".format(now.day), "a") as f:
            f.write(str('#'*35)+  str(name) + str('#'*35) + '\n')
            f.write('\n')
        for n, clf in zip(names, classifiers):
            with open(log_file + "{}_".format(now.year)+"{}_".format(now.month)+"{}".format(now.day), "a") as f:
                f.write(str('*'*30) + str(n) + str('*'*30) + '\n')
                f.write(str(clf) + "\n")

# ####################Feature Processing####################
# Including 1. Imputation,
#           2. Standardization,
#           3. Normalizaiton
# ##########################################################

def feature_processing(names,preprocessors,features,test_feature):
    progress_log(names, preprocessors)
    for name, preprocessor in zip(names,preprocessors):
        print("Start {}".format(name))
        preprocessor.fit(features)
        features = preprocessor.transform(features)
        test_feature = preprocessor.transform(test_feature)
    return features, test_feature

# ####################Feature Selection######################
def select_features_from_xgb(features,labels,test_feature):

    print("\nStart selecting importance features")
    xgb = XGBClassifier(n_estimators=2, max_depth=4, learning_rate = 0.07, subsample = 0.8, colsample_bytree = 0.9)
    xgb = xgb.fit(features, labels)
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]

    model = SelectFromModel(xgb, prefit=True)
    features_new = model.transform(features)
    test_feature_new = model.transform(test_feature)
    with open(data_path + "importance_features.txt" , "w") as log:
        for f in range(features_new.shape[1]):
            log.write(str(f + 1) + "." +  " feature " +  str(indices[f]) + "  " + str(importances[indices[f]]) + "\n")
            #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print("Features selection done saved new data in data path")
    
    sel = VarianceThreshold(threshold = (.8 * (1 - .8)))
    sel.fit_transform(features)
    
    return features_new, test_feature_new

# ####################CV Slicing############################
def stack_split(features, labels, number_of_model):
    # Define number of sizes per model
    fold_size = int(labels.size/number_of_model)

    # Iterate number of models to get different fold, feature and label data
    fold_split = {}
    fold_split_label = {}
    feature_split = {}
    label_split = {}

    for i in range(number_of_model):
        # define starting and end rows of the fold data
        start_row = fold_size * i
        end_row = fold_size * (i+1)

        if i == number_of_model - 1:

            print("\nfold_{}".format(i+1) + " starting between row:{}".format(start_row) + " and row:{}".format(end_row))
            fold_split["fold_{}".format(i+1)] = features[start_row:,:]
            fold_split_label["fold_label_{}".format(i+1)] = labels[start_row:]
            # Delete the extrated data from feature and label data
            feature_split["feature_{}".format(i+1)] = np.delete(features, np.s_[start_row:], axis = 0)
            label_split["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:], axis = 0)

        else:

            print("\nfold_{}".format(i+1) + " starting between row:{}".format(start_row) + " and row:{}".format(end_row))
            # Store extrated fold data from feature
            fold_split["fold_{}".format(i+1)] = features[start_row:end_row,:]
            fold_split_label["fold_label_{}".format(i+1)] = labels[start_row:end_row]
            # Delete the extrated data from feature and label data
            feature_split["feature_{}".format(i+1)] = np.delete(features, np.s_[start_row:(start_row + fold_size)], axis = 0)
            label_split["label_{}".format(i+1)] = np.delete(labels, np.s_[start_row:(start_row + fold_size)], axis = 0)

    return fold_split, fold_split_label, feature_split, label_split

def save_final_layer_score(score):
    final_test_path = sub_log_file + "{}_{}_{}:{}.csv".format(now.month, now.day, now.hour, now.minute)
    final_score = pd.read_csv(stack_test_path)
    f_score = final_score.assign(score = score)
    f_score.to_csv(final_test_path, index = None, float_format = "%.9f")
    progress_log(end_log = True)
    print("\nFinal score saved to {}".format(final_test_path))

def stack_xgb(features, labels, test, unlabel):

    dtrain = xgb.DMatrix(features, label=labels)
    dtest = xgb.DMatrix(test)
    dunlabel = xgb.DMatrix(unlabel)
    bst = xgb.train(param, dtrain, num_round)
    bst.save_model(model_path + "XGB_layer_2.model")
    print("\nSaved model <XGB_layer_2.model>")
    final_preds = bst.predict(dtest)
    ublabel_preds = bst.predict(dunlabel)
    np.savetxt(sub_log_file + "{}_unlabel_{}:{}.csv".format("final", now.hour, now.minute) ,ublabel_preds ,fmt = '%.9f', delimiter = ',')

    return final_preds, ublabel_preds

def stack_layer(names, classifiers, features, labels, test_feature, unlabel, layer_name):

        progress_log(names, classifiers, layer_name)
        fold_split, fold_split_label, feature_split, label_split = stack_split(features,labels,5)
        layer_transform_train = []
        layer_transform_test = []
        layer_transform_unlabel = []
        for name, clf in zip(names, classifiers):
            fold_score = []
            test_score = []
            unlabel_score = []
            plt.figure()
            for i in range(len(fold_split)):
                start = time.time()
                print("\nProcessing model :{} fold {}".format(name, i+1))
                clf.fit(feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)])
                print("Training complete")
                stack_score = clf.predict_proba(fold_split["fold_{}".format(i+1)])
                print("fold score predicted")
                test_prediction = clf.predict_proba(test_feature)
                print("test score predicted")
                unlabel_prediction = clf.predict_proba(unlabel)
                print("unlabel score predicted")
                unlabel_score.append(unlabel_prediction[:,1].tolist())
                test_score.append(test_prediction[:,1].tolist())
                fold_score += stack_score[:,1].tolist()
                print("model {}".format(name) + " complete")
                #scores = model_selection.cross_val_score(clf, feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)],
                #       cv=5, scoring= "roc_auc")
                #y_probs = model_selection.cross_val_predict(clf, feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)],
                       #cv=3, method= "predict_proba")
                #f1_scores = model_selection.cross_val_score(clf, feature_split["feature_{}".format(i+1)], label_split["label_{}".format(i+1)],
                       #cv=5, scoring='f1')
                #print("AUC: %0.2f (+/- %0.2f) [%s] in fold %i" % (scores.mean(), scores.std(), name, i))
                fpr, tpr, thresholds = roc_curve(fold_split_label["fold_label_{}".format(i+1)], stack_score[:,1])
                plot_roc_curve(fpr, tpr, name+"_{}".format(i+1))
                roc_score = roc_auc_score(fold_split_label["fold_label_{}".format(i+1)], stack_score[:,1])
                print("ROC score: {}".format(roc_score))
                #plt.text(0.0085, 0.1 ,str(round(roc_score,4)), fontsize = 10)
                plt.savefig(sub_log_file + "{}_{}_{}_auc:{}.png".format(layer_name, name, i+1, round(roc_score,4)))
                end = time.time()
                print(">>>>Duration<<<< : {}min ".format(round((end-start)/60,2)))

            #Averaging stacked
            stack_test_layer1_preds = np.stack(test_score, 1)
            stack_unlabel_layer1_preds = np.stack(unlabel_score, 1)
            #averaging stacked data
            avged_test_preds = []
            avged_unlabel_preds = []
            for row in stack_test_layer1_preds:
                avg = np.mean(row)
                avged_test_preds.append(avg)
            print("\nAveraging test score done ......")

            for row in stack_unlabel_layer1_preds:
                avg = np.mean(row)
                avged_unlabel_preds.append(avg)
            print("\nAveraging unlabel score done ......")

            layer_transform_train.append(fold_score)
            layer_transform_test.append(avged_test_preds)
            layer_transform_unlabel.append(avged_unlabel_preds)

        layer_transform_train = np.array(layer_transform_train).transpose()
        layer_transform_test = np.array(layer_transform_test).transpose()
        layer_transform_unlabel = np.array(layer_transform_unlabel).transpose()
        np.savetxt(sub_log_file + "{}_train_{}:{}.csv".format(layer_name, now.hour, now.minute) ,layer_transform_train , fmt = '%.9f', delimiter = ',')
        np.savetxt(sub_log_file + "{}_test_{}:{}.csv".format(layer_name, now.hour, now.minute) ,layer_transform_test ,fmt = '%.9f', delimiter = ',')
        np.savetxt(sub_log_file + "{}_unlabel_{}:{}.csv".format(layer_name, now.hour, now.minute) ,layer_transform_unlabel ,fmt = '%.9f', delimiter = ',')

        return layer_transform_train, layer_transform_test, layer_transform_unlabel

def k_means_uncertain(unlabel, unlabel_preds):
    unlabel_preds = unlabel_preds.tolist()
    uncertain_index = [unlabel_preds.index(i) for i in unlabel_preds if i > 0.3 and i < 0.6]
    unlabel = unlabel.tolist()
    uncertain = np.array([unlabel[i] for i in uncertain_index])
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit(uncertain)
    #print(len(kmeans.labels_))
   

def main():
    warnings.filterwarnings(module = 'sklearn*', action = 'ignore', category = DeprecationWarning)
    progress_log(start_log = True)

    train_data = np.load(train_path)
    test = np.load(test_path)
    unlabel = np.load(unlabel_path)

    features = train_data[:,1:]
    label = train_data[:,0]
    """
    prepro_names = ["Imputation", "StandardScaler", "Normalizaiton"]

    preprocessors = [
        Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        StandardScaler()
        Normalizer(norm='l2')
    ]
    """

    #features, test = feature_processing(prepro_names, preprocessors, features, test_feature)

    #features, test = select_features_from_xgb(features, label, test)

    # ####################First Layer Start#####################
    clf_names = ["XGB", "RF", "LR", "Ada"]
    classifier = [

        XGBClassifier(n_estimators=450, max_depth=4, learning_rate = 0.06,
                      gamma = 0.1, min_child_weight = 2, reg_alpha = 0.05,
                      subsample = 0.6, colsample_bytree = 0.8, n_jobs = -1),

        RandomForestClassifier(n_estimators = 450, max_depth = 5, 
                               criterion='entropy', min_samples_split = 2, n_jobs = -1), #450
        #MLPClassifier(hidden_layer_sizes=(128,64,32), activation = "logistic", batch_size = 20000),
        LogisticRegression(class_weight = "balanced"),
        #AdaBoostClassifier(n_estimators = 400, learning_rate = 0.08),

    ]

    features, test, unlabel = stack_layer(clf_names, classifier, features, label, test, unlabel, layer_name = "layer1")

    # ####################Second Layer Start#####################
    layer2_clf_names = ["XGB", "ET", "LR"]

    layer2_classifier = [
        XGBClassifier(n_estimators=450, max_depth=3, learning_rate = 0.06,
                          gamma = 0.1,
                          subsample = 0.6, colsample_bytree = 1, n_jobs = -1),
        #KNeighborsClassifier(n_neighbors=5, weights='uniform', leaf_size=30, n_jobs=-1),
        #QDA(),
        ExtraTreesClassifier(n_estimators = 400, max_depth = 3, criterion='entropy'),
        LogisticRegression(class_weight = "balanced"),
        #MLPClassifier(hidden_layer_sizes=(256,128,128), activation = "logistic", batch_size = 20000)
        #RandomForestClassifier(n_estimators = 3, max_depth = 4, criterion='entropy', n_jobs = -1), #450
    ]

    features, test, unlabel = stack_layer(layer2_clf_names, layer2_classifier, features, label, test, unlabel, layer_name = "layer2")

    final_preds, unlabel_preds = stack_xgb(features, label, test, unlabel)

    save_final_layer_score(final_preds)

    k_means_uncertain(unlabel, unlabel_preds)


if __name__ == '__main__':
    main()
