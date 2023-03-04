import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import glob

from utils.utils import remove_infs, make_labels_binary, subset_normal, split_in_batches

if __name__ =='__main__':
    df = pd.read_csv('../data/cicids2017_kaggle/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
    df = df.rename(columns={' Label': 'Label'})

    before_removal = len(df)
    df, labels = remove_infs(df)
    print(f'Length before NaN drop: {before_removal}, after NaN drop: {len(df)}\n \
    The df is now {len(df)/before_removal} of its original size')

    le = LabelEncoder()
    le.fit(labels)

    int_labels = le.transform(labels)

    train_ratio = 0.65
    val_ratio = 0.15
    test_ratio = 0.2
    x_train, x_test, y_train, y_test = train_test_split(df, int_labels, test_size=1-train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + val_ratio))
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == x_test.shape[1]
    assert x_val.shape[0] == y_val.shape[0]
    assert x_val.shape[1] ==  x_test.shape[1]

    y_train = make_labels_binary(le, y_train)
    y_val = make_labels_binary(le, y_val)
    y_test = make_labels_binary(le, y_test)

    # Subsetting only Normal Network packets in training set
    x_train, y_train = subset_normal(x_train, y_train)
    x_val, y_val = subset_normal(x_val, y_val)
    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.to_numpy().astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    preprocessed_data = {
        'dataset': dataset,
    }

    with open('../data/preprocessed_data.pickle', 'wb') as file:
        pickle.dump(preprocessed_data, file)