import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import glob

from utils.utils import DataSequence
from utils.utils import remove_infs, make_labels_binary, subset_normal

BATCH_SIZE = 256

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
    
    train_x, test_x, train_y, test_y = train_test_split(df, int_labels, test_size=0.2)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == x_test.shape[1]
    assert x_val.shape[0] == y_val.shape[0]
    assert x_val.shape[1] ==  x_test.shape[1]

    y_train = make_labels_binary(le, y_train)
    y_val = make_labels_binary(le, y_val)
    y_test = make_labels_binary(le, y_test)

    train_y = make_labels_binary(le, train_y)
    test_y = make_labels_binary(le, test_y)
    # Subsetting only Normal Network packets in training set
    x_train, y_train = subset_normal(x_train, y_train)
    x_val, y_val = subset_normal(x_val, y_val)

    train_x, train_y = subset_normal(train_x, train_y)
    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    scaler.transform(train_x)
    train_sqc = DataSequence(x_train, y_train, batch_size=BATCH_SIZE)
    val_sqc = DataSequence(x_val, y_val, BATCH_SIZE)
    test_sqc = DataSequence(x_test, y_test, BATCH_SIZE)

    train_set = DataSequence(train_x, train_y, batch_size=BATCH_SIZE)
    test_set = DataSequence(test_x, test_y, batch_size=BATCH_SIZE)

    dataset = {}
    dataset['train'] = train_sqc
    dataset['val'] = val_sqc
    dataset['test'] = test_sqc

    
    test_dataset = {}
    test_dataset['train'] = train_set
    test_dataset['test']  = test_set
    preprocessed_data = {
        'dataset': dataset,
        'test_dataset': test_dataset
    }

    with open('../data/preprocessed_data.pickle', 'wb') as file:
        pickle.dump(preprocessed_data, file)