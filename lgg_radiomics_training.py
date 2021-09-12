data_dir = 'dataset/'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

df_trn = pd.read_csv(os.path.join(data_dir, 'dataset', 'LGG.radiomics.trn.csv'))
df_tst = pd.read_csv(os.path.join(data_dir, 'dataset', 'LGG.radiomics.tst.csv'))

X_trn = df_trn.drop('Label', axis=1)
y_trn = df_trn['Label']
X_tst = df_tst.drop('Label', axis=1)
y_tst = df_tst['Label']

"""## Machine Learning"""

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from sklearn import metrics
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

def mlp_model():
    clf = MLPClassifier()
    return clf

def logicstic_regression():
    clf = LogisticRegression()
    return clf

def knn_model():
    knn = KNeighborsClassifier(n_neighbors=10)
    return knn

def randomforest_model():
    rf = RandomForestClassifier(bootstrap=True, max_depth=80, max_features='sqrt',
                       min_samples_leaf=2, min_samples_split=10,n_estimators=300)
    return rf

def xgboost_model():
    xg = xgb.XGBClassifier(booster='gbtree', colsample_bytree=0.4, gamma=1,
              learning_rate=0.9, max_depth=4, min_child_weight=10, n_estimators=700, 
              subsample=0.8)
    return xg

"""### Baseline model comparison"""

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
kfold = StratifiedKFold(n_splits=5, shuffle=True)
# Comparison of the performance results among different baseline models
lr_model = LogisticRegression(max_iter=1000)
lr_result = cross_val_score(lr_model, X_trn, y_trn, cv=kfold)
print('Logistic Regression: ', lr_result.mean())

rf_model = RandomForestClassifier()
rf_result = cross_val_score(rf_model, X_trn, y_trn, cv=kfold)
print('Random Forest: ', rf_result.mean())

svm_model = SVC()
svm_result = cross_val_score(svm_model, X_trn, y_trn, cv=kfold)
print('Support Vector Machine: ', svm_result.mean())

ab_model = AdaBoostClassifier()
ab_result = cross_val_score(ab_model, X_trn, y_trn, cv=kfold)
print('AdaBoost: ', ab_result.mean())

xgb_model = xgb.XGBClassifier()
xgb_result = cross_val_score(xgb_model, X_trn, y_trn, cv=kfold)
print('XGBoost: ', xgb_result.mean())

"""## Cross-validation"""

kfold = StratifiedKFold(n_splits=5, shuffle=True)
TP = FP = TN = FN = 0
acc_cv_scores = []
auc_cv_scores = []

for train, test in kfold.split(X_trn, y_trn):
    svm_model = xgboost_model()   
    ## evaluate the model
    svm_model.fit(X_trn.iloc[train], y_trn.iloc[train])
    # evaluate the model
    true_labels = np.asarray(y_trn.iloc[test])
    predictions = svm_model.predict(X_trn.iloc[test])
    acc_cv_scores.append(accuracy_score(true_labels, predictions))
    newTN, newFP, newFN, newTP = confusion_matrix(true_labels,predictions).ravel()
    TP += newTP
    FN += newFN
    FP += newFP
    TN += newTN
    pred_prob = svm_model.predict_proba(X_trniloc[test])
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_prob[:,1], pos_label=1)
    auc_cv_scores.append(metrics.auc(fpr, tpr))

print('Accuracy = ', np.mean(acc_cv_scores))
print('TP = %s, FP = %s, TN = %s, FN = %s' % (TP, FP, TN, FN))
print('AUC = ', np.mean(auc_cv_scores))

"""### Imbalance"""

from imblearn.over_sampling import SMOTE
from collections import Counter

ros = SMOTE()
TP = FP = TN = FN = 0
acc_cv_scores = []
auc_cv_scores = []

for train, test in kfold.split(X_trn, y_trn):
    train_x, train_y = X_trn.iloc[train], y_trn.iloc[train]
    test_x, test_y = X_trn.iloc[test], y_trn.iloc[test]
    test_x = np.asarray(test_x)
    X_ros, y_ros = ros.fit_resample(train_x, train_y)
    svm_model = xgboost_model()   
    ## evaluate the model
    svm_model.fit(X_ros, y_ros)
    # evaluate the model
    true_labels = np.asarray(y_trn.iloc[test])
    predictions = svm_model.predict(test_x)
    acc_cv_scores.append(accuracy_score(true_labels, predictions))
    newTN, newFP, newFN, newTP = confusion_matrix(true_labels,predictions).ravel()
    TP += newTP
    FN += newFN
    FP += newFP
    TN += newTN
    pred_prob = svm_model.predict_proba(test_x)
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_prob[:,1], pos_label='yes')
    auc_cv_scores.append(metrics.auc(fpr, tpr))

print('Accuracy = ', np.mean(acc_cv_scores))
print('TP = %s, FP = %s, TN = %s, FN = %s' % (TP, FP, TN, FN))
print('AUC = ', np.mean(auc_cv_scores))


# External Validation Test
final_model = xgboost_model()
final_model.fit(X_trn, y_trn)
true_labels = np.asarray(y_tst)
predictions = final_model.predict(X_tst)
print(accuracy_score(true_labels, predictions))
print(confusion_matrix(true_labels, predictions))