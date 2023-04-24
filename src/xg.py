from pathlib import Path
import xgboost as xgb
import numpy as np
from collections import OrderedDict
import gc
from glob import glob
import os
import pandas as pd
from copy import copy
from time import time
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score,classification_report,roc_curve
import json
from time import time

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score
from utils.plots import  plot_roc, plot_precision_recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

from utils.utils import calc_metrics
from utils.plots import plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats

FILENAME = '../data/preprocessed_data.pickle'

def calc_all(model, test):
    threshold = .5
    true_labels = test.y.astype(int)

    preds = model.predict(test.x)
    pred_labels = (preds > threshold).astype(int)
    accuracy = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = confusion_matrix(true_labels, pred_labels, normalize='all')
    precision, recall, f1, _ = precision_recall_fscore_support(test.y, pred_labels, average='binary')
    accuracy = accuracy_score(test.y, pred_labels)
    fpr, tpr, thresholds = roc_curve(test.y, preds)
    auc_val = auc(fpr, tpr)

    metrics = calc_metrics(cm)
    metrics['f1'] = f1
    metrics['AUC'] = auc_val
    d = dict(metrics)

    return d, cm, cm_norm, preds

def xg_main(train, test, frags, trials, save='xg'):
    name = '../experiments/' + save + '/best/'
    Path(name).mkdir(parents=True, exist_ok=True)
    assert train.x.shape[0] == train.y.shape[0]
    assert test.x.shape[0] == test.y.shape[0]
    assert test.x.shape[1] == train.x.shape[1]
    train.y = train.y.astype(int)
    
    params = {
        'num_rounds':        10,
        'max_depth':         8,
        'max_leaves':        2**8,
        'alpha':             0.9,
        'eta':               0.1,
        'gamma':             0.1,
        'learning_rate':     0.1,
        'subsample':         1,
        'reg_lambda':        1,
        'scale_pos_weight':  2,
        'objective':         'binary:logistic',
        'verbose':           True,
        'gpu_id':            0,
        'tree_method':       'gpu_hist'
    }
    hyperparameter_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.20],
        'max_leaves': [2**4, 2**6, 2**8]
    }
    #dtrain = xgb.DMatrix(train.x, label=train.y, feature_weights=feature_weights)
    #dtest = xgb.DMatrix(test.x, label=test.y, feature_weights=feature_weights)
    bst = xgb.XGBClassifier(**params)
    clf = RandomizedSearchCV(bst, hyperparameter_grid, random_state=0, n_iter=trials)
    
    start = time()
    model = clf.fit(train.x, train.y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - start, len(clf.cv_results_["params"])) )  
    print(clf.cv_results_)

    print(model.best_params_)

    # threshold = .5
    # true_labels = test.y.astype(int)
    # preds = model.predict(test.x)
    # pred_labels = (preds > threshold).astype(int)
    # auc_x = roc_auc_score(true_labels, preds)
    # accuracy = accuracy_score(true_labels, pred_labels)
    # cm = confusion_matrix(true_labels, pred_labels)
    # cm_norm = confusion_matrix(true_labels, pred_labels, normalize='all')

    # precision, recall, f1, _ = precision_recall_fscore_support(test.y, pred_labels, average='binary')
    # accuracy = accuracy_score(test.y, pred_labels)
    # fpr, tpr, thresholds = roc_curve(test.y, preds)
    # auc_val = auc(fpr, tpr)
    
    if frags is not None:
        metrics_frag, cm_frag, cm_frag_norm, preds_frag = calc_all(model, frags)

    # precision, recall, f1, _ = precision_recall_fscore_support(test.y, pred_labels, average='binary')
    # accuracy = accuracy_score(test.y, pred_labels)
    # fpr, tpr, thresholds = roc_curve(test.y, preds)
    # auc_val = auc(fpr, tpr)
    
    metrics, cm, cm_norm, preds = calc_all(model, test)
    plot_confusion_matrix(cm, savefile=name + save + '_cm.pdf', name=save)
    plot_confusion_matrix(cm_norm, savefile=name + save + '_cm_normalized.pdf', name=save)
    plot_roc(metrics['TPR'], metrics['FPR'], metrics['AUC'], name + save + '_roc.pdf', name=save)
    preds = np.vstack((1-preds, preds)).T
    plot_precision_recall(test.y, preds, name + save + '_precision_recall.pdf')
    
    
    results = {
            'Metrics': metrics,
            'Metrics frag': metrics_frag
    }
    print('End results', results)
    with open('../experiments/' + save + '/best/' + save + '_best_model.json', 'w', encoding='utf-8') as f: 
        json.dump(results, f, ensure_ascii=False, indent=4)

    model.best_estimator_.save_model('models/' + save + '.model')
    return model.best_estimator_
