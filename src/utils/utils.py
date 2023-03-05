import json
import math

import pandas as pd
import numpy as np

import tensorflow as tf
# Labels normal data as 0, anomalies as 1
def make_labels_binary(label_encoder, labels):
    normal_data_index = np.where(label_encoder.classes_ == 'BENIGN')[0][0]
    new_labels = labels.copy()
    new_labels[labels != normal_data_index] = 1
    new_labels[labels == normal_data_index] = 0
    return new_labels

def subset_normal(x_train, y_train):
    temp_df = x_train.copy()
    temp_df['label'] = y_train
    temp_df = temp_df.loc[temp_df['label'] == 0]
    y_train = temp_df['label'].copy()
    temp_df = temp_df.drop('label', axis = 1)
    x_train = temp_df.copy()
    return x_train,y_train

def save_results(name, config, results):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        json.dump(results, f, ensure_ascii=False, indent=4)

def reduce_anomalies(df, pct_anomalies=.01):
    labels = df['Label'].copy()
    is_anomaly = labels != 'BENIGN'
    num_normal = np.sum(~is_anomaly)
    num_anomalies = int(pct_anomalies * num_normal)
    all_anomalies = labels[labels != 'BENIGN']
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