import json
import math

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
# Labels normal data as 0, anomalies as 1
def make_labels_binary(label_encoder, labels):
    print('LABEL ENCODER', np.where(label_encoder.classes_ == 'Benign'))
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

def scale(x_train, x_test):
    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test

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
    FP = FP[1]
    FN = FN[1]
    TP = TP[1]
    TN = TN[1]

    print('TP', TP)
    print('TN', TN)
    print('FP', FP)
    print('FN', FN)
    met['TP'] = int(TP)
    met['TN'] = int(TN)
    met['FP'] = int(FP)
    met['FN'] = int(FN)

    met['TPR'] = (TP/(TP+FN)).tolist()
    met['TNR'] = (TN/(TN+FP)).tolist()
    met['PPV'] = (TP/(TP+FP)).tolist()
    met['NPV'] = (TN/(TN+FN)).tolist()
    met['FPR'] = (FP/(FP+TN)).tolist()
    met['FNR'] = (FN/(TP+FN)).tolist()
    met['FDR'] = (FP/(TP+FP)).tolist()

    met['ACC'] = ((TP+TN)/(TP+FP+FN+TN)).tolist()
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