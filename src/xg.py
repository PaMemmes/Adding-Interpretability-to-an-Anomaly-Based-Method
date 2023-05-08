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
import json
from time import time

from utils.plots import  plot_roc, plot_precision_recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

from utils.utils import calc_all
from utils.plots import plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats

FILENAME = '../data/preprocessed_data.pickle'

def xg_main(train, test, frags, trials, save='xg'):
    name = '../experiments/' + save + '/best/'
    Path(name).mkdir(parents=True, exist_ok=True)
    
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
        'max_leaves': [2**4, 2**6, 2**8],
        'eta': [x for x in np.linspace(0.1, 0.6, 6)],
        'gamma': [int(x) for x in np.linspace(0, 0.5, 6)]
    }

    bst = xgb.XGBClassifier(**params)
    clf = RandomizedSearchCV(bst, hyperparameter_grid, random_state=0, n_iter=trials)
    
    start = time()
    model = clf.fit(train.x, train.y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - start, len(clf.cv_results_["params"])))  
    print(model.best_params_)

    metrics_train, cm_train, cm_norm_train, preds_train = calc_all(model, train)
    plot_confusion_matrix(cm_train, savefile=name + save + '_cm_train.pdf', name=save)
    plot_confusion_matrix(cm_norm_train, savefile=name + save + '_cm_normalized_train.pdf', name=save)
    plot_roc(metrics_train['TPR'], metrics_train['FPR'], metrics_train['AUC'], name + save + '_roc_train.pdf', name=save)
    preds_train = np.vstack((1-preds_train, preds_train)).T

    metrics, cm, cm_norm, preds = calc_all(model, test)
    plot_confusion_matrix(cm, savefile=name + save + '_cm.pdf', name=save)
    plot_confusion_matrix(cm_norm, savefile=name + save + '_cm_normalized.pdf', name=save)
    plot_roc(metrics['TPR'], metrics['FPR'], metrics['AUC'], name + save + '_roc.pdf', name=save)
    preds = np.vstack((1-preds, preds)).T
    # plot_precision_recall(test.y, preds, name + save + '_precision_recall.pdf')
    
    metrics_frag, cm_frag, cm_frag_norm, preds_frag = calc_all(model, frags)
    plot_confusion_matrix(cm_frag, savefile=name + save + '_frags_cm.pdf', name=save)
    plot_confusion_matrix(cm_frag_norm, savefile=name + save + '_frags_cm_normalized.pdf', name=save)
    plot_roc(metrics_frag['TPR'], metrics_frag['FPR'], metrics_frag['AUC'], name + save + '_frags_roc.pdf', name=save)
    preds_frag = np.vstack((1-preds_frag, preds_frag)).T
    #plot_precision_recall(frags.y, preds_frag, name + save + '_frags_precision_recall.pdf')

    
    results = {
            'Metrics': metrics,
            'Metrics frag': metrics_frag,
            'Metrics_train': metrics_train,
            'Best hyperparameters': model.best_params_
    }
    
    print('End results', results)
    with open('../experiments/' + save + '/best/' + save + '_best_model.json', 'w', encoding='utf-8') as f: 
        json.dump(results, f, ensure_ascii=False, indent=4)

    model.best_estimator_.save_model('models/' + save + '.model')
    return model.best_estimator_
