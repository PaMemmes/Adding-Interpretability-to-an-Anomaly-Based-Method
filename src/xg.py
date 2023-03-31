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

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score
from utils.plots import  plot_roc, plot_precision_recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.utils import calc_metrics
from utils.plots import plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle

FILENAME = '../data/preprocessed_data.pickle'


# def plot_confusion_matrix(cm, target_names, save, title='Confusion Matrix', cmap=plt.cm.Greens):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(target_names))
#     plt.xticks(tick_marks, target_names, rotation=45)
#     plt.yticks(tick_marks, target_names)
#     plt.tight_layout()

#     width, height = cm.shape

#     for x in range(width):
#         for y in range(height):
#             plt.annotate(str(cm[x][y]), xy=(y, x), 
#                         horizontalalignment='center',
#                         verticalalignment='center')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.savefig(save + 'cm.png')
#     plt.close()


def xg_main(save='xg'):
    name = '../experiments/' + save + '/best/'
    input_file = open(FILENAME, 'rb')
    preprocessed_data = pickle.load(input_file)
    input_file.close()

    train_dataset = preprocessed_data['dataset']

    num_features = train_dataset['train'].x.shape[1]
    train = train_dataset['train']
    test = train_dataset['test']
    
    assert train.x.shape[0] == train.y.shape[0]
    assert test.x.shape[0] == test.y.shape[0]
    assert test.x.shape[1] == train.x.shape[1]

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
        'verbose':           True
    }
    dtrain = xgb.DMatrix(train.x, label=train.y, feature_weights=[0.9,0.1])
    dtest = xgb.DMatrix(test.x, label=test.y, feature_weights=[0.9,0.1])
    evals = [(dtest, 'test',), (dtrain, 'train')]
    num_rounds = params['num_rounds']
    model = xgb.train(params, dtrain, num_rounds,evals=evals)
    threshold = .5
    true_labels = test.y.astype(int)
    true_labels.sum()
    preds = model.predict(dtest)
    pred_labels = (preds > threshold).astype(int)
    auc_x = roc_auc_score(true_labels, preds)
    accuracy = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels) 

    precision, recall, f1, _ = precision_recall_fscore_support(test.y, pred_labels, average='binary')
    accuracy = accuracy_score(test.y, pred_labels)
    fpr, tpr, thresholds = roc_curve(test.y, preds)
    auc_val = auc(fpr, tpr)
    plot_confusion_matrix(cm, savefile=name+'xg.png', name='xg')
    plot_roc(tpr, fpr, auc_val, name + 'roc_gan_only_cic.png', 'GAN')
    preds = np.vstack((1-preds, preds)).T
    plot_precision_recall(test.y, preds, name + 'precision_recall_only_cic.png')
    results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc_x
    }
    print(results)
    metrics = calc_metrics(cm)
    
    d = dict(metrics)
    print(d)
    with open('../experiments/' + save + '/best/best_model.json', 'w', encoding='utf-8') as f: 
        json.dump(results, f, ensure_ascii=False, indent=4)
        json.dump(d, f, ensure_ascii=False, indent=4)

    model.save_model('models/' + save + '.model')
    return model
