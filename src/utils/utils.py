import json
import math

import collections
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score,confusion_matrix,accuracy_score,classification_report,roc_curve
import glob
import os

import tensorflow as tf

def read_csv():
    #all_files = glob.glob(os.path.join('../data/cicids2018', "*.csv"))
    #df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    df = pd.read_csv('../data/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
    return df

def make_labels_binary(label_encoder, labels):
    normal_data_index = np.where(label_encoder.classes_ == 'Benign')[0][0]
    new_labels = labels.copy()
    new_labels[labels != normal_data_index] = 1
    new_labels[labels == normal_data_index] = 0
    return new_labels

# Normal is 0, anomaly is 1
def subset(x_train, y_train, kind=0):
    temp_df = x_train.copy()
    temp_df['Label'] = y_train
    temp_df = temp_df.loc[temp_df['Label'] == kind]
    y_train = temp_df['Label'].copy()
    temp_df = temp_df.drop('Label', axis = 1)
    x_train = temp_df.copy()
    return x_train, y_train

def save_results(name, config, results):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        json.dump(results, f, ensure_ascii=False, indent=4)

def reduce_anomalies(df, pct_anomalies=.01):
    labels = df['Label'].copy()
    is_anomaly = labels != 'Benign'
    num_normal = np.sum(~is_anomaly)
    num_anomalies = int(pct_anomalies * num_normal)
    all_anomalies = labels[labels != 'Benign']
    anomalies_to_keep = np.random.choice(
        all_anomalies.index, size=num_anomalies, replace=False)
    anomalous_data = df.iloc[anomalies_to_keep].copy()
    normal_data = df[~is_anomaly].copy()
    new_df = pd.concat([normal_data, anomalous_data], axis=0)
    return new_df

# Remove infinities and NaNs
def remove_infs(df):
    assert isinstance(df, pd.DataFrame)
    labels = df['Label']
    df = df.drop('Label', axis=1)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep], labels[indices_to_keep]

def encode(le, labels):
    labels = le.transform(labels)

    return labels

def open_config(model_name):
    with open('configs/config_' + model_name + '.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
    return config

def get_preds(results, train):

    normals = collections.Counter(train.y)[0]
    anomalies = collections.Counter(train.y)[1]
    anomalies_percentage = anomalies / (normals + anomalies)
    
    # Obtaining the lowest "anomalies_percentage" score
    per = np.percentile(results, anomalies_percentage*100)
    results = np.array(results)
    y_pred = results.copy()
    probas = np.vstack((1-y_pred, y_pred)).T

    inds = y_pred > per
    inds_comp = y_pred <= per
    y_pred[inds] = 0
    y_pred[inds_comp] = 1
    return y_pred, probas, per, anomalies_percentage

def calc_all_nn(test, y_pred, probas):
    precision, recall, f1, _ = precision_recall_fscore_support(test.y, y_pred, average='binary')
    accuracy = accuracy_score(test.y, y_pred)
    fpr, tpr, thresholds = roc_curve(test.y, probas[:,0])
    
    auc_val = auc(fpr, tpr)
    cm = confusion_matrix(test.y, y_pred)
    cm_norm = confusion_matrix(test.y, y_pred, normalize='all')
    metrics = calc_metrics(cm)
    _, _, metrics['f1'], _ = precision_recall_fscore_support(test.y, y_pred, average='binary')
    metrics['AUC'] = auc_val
    d = dict(metrics)

    return d, cm, cm_norm


def calc_all(model, test):
    threshold = .5
    true_labels = test.y.astype(int)

    preds = model.predict(test.x)
    pred_labels = (preds > threshold).astype(int)
    accuracy = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = confusion_matrix(true_labels, pred_labels, normalize='all')
    _, _, f1, _ = precision_recall_fscore_support(test.y, pred_labels, average='binary')
    accuracy = accuracy_score(test.y, pred_labels)
    fpr, tpr, thresholds = roc_curve(test.y, preds)
    auc_val = auc(fpr, tpr)

    metrics = calc_metrics(cm)
    metrics['f1'] = f1
    metrics['AUC'] = auc_val
    d = dict(metrics)

    return d, cm, cm_norm, preds

def test_model(model, test):
    nr_batches_test = np.ceil(test.x.shape[0] // test.batch_size).astype(np.int32)
    results = []
    for t in range(nr_batches_test + 1):
        ran_from = t * test.batch_size
        ran_to = (t + 1) * test.batch_size
        image_batch = test.x[ran_from:ran_to]
        tmp_rslt = model.discriminator.predict(x=image_batch, batch_size=test.batch_size, verbose=0)
        results = np.append(results, tmp_rslt)
    pd.options.display.float_format = '{:20,.7f}'.format
    results_df = pd.concat([pd.DataFrame(results), pd.DataFrame(test.y)], axis=1)
    results_df.columns = ['results', 'y_test']
    return results_df, results

def calc_metrics(confusion_matrix):
    met = defaultdict()
    
    FP = (confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))
    FN = (confusion_matrix.sum(axis=1) - np.diag(confusion_matrix))
    TP = np.diag(confusion_matrix)
    TN = (confusion_matrix.sum() - (FP + FN + TP))
 
    if len(FP) == 2:
        FP = FP[1]
        FN = FN[1]
        TP = TP[1]
        TN = TN[1]
    else:
        FP = FP[0]
        FN = FN[0]
        TP = TP[0]
        TN = TN[0]

    met['BACC'] = ((TP/(TP+FN)).tolist() + (TN/(TN+FP)).tolist()) / 2
    met['ACC'] = ((TP+TN)/(TP+FP+FN+TN)).tolist()

    met['TPR'] = (TP/(TP+FN)).tolist()
    met['TNR'] = (TN/(TN+FP)).tolist()
    met['PPV'] = (TP/(TP+FP)).tolist()
    met['NPV'] = (TN/(TN+FN)).tolist()
    met['FPR'] = (FP/(FP+TN)).tolist()
    met['FNR'] = (FN/(TP+FN)).tolist()
    met['FDR'] = (FP/(TP+FP)).tolist()

    met['TP'] = int(TP)
    met['TN'] = int(TN)
    met['FP'] = int(FP)
    met['FN'] = int(FN)

    return met

class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set.astype(np.float32), y_set.astype(np.float32)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array(batch_x), np.array(batch_y)