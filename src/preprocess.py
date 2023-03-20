import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import glob

from utils.utils import DataSequence
from utils.utils import remove_infs, make_labels_binary, subset_normal

BATCH_SIZE = 256

def preprocess(subset=True):
    df = pd.read_csv('../data/cicids2017_kaggle/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
    df = df.rename(columns={' Label': 'Label'})

    before_removal = len(df)
    df, labels = remove_infs(df)
    print(f'Length before NaN drop: {before_removal}, after NaN drop: {len(df)}\nThe df is now {len(df)/before_removal} of its original size')

    le = LabelEncoder()
    le.fit(labels)

    int_labels = le.transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(df, int_labels, test_size=0.2)

    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == x_test.shape[1]

    y_train = make_labels_binary(le, y_train)
    y_test = make_labels_binary(le, y_test)

    # Subsetting only Normal Network packets in training set
    if subset is True:
        x_train, y_train = subset_normal(x_train, y_train)

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    train_sqc = DataSequence(x_train, y_train, batch_size=BATCH_SIZE)
    test_sqc = DataSequence(x_test, y_test, BATCH_SIZE)

    dataset = {}
    dataset['train'] = train_sqc
    dataset['test'] = test_sqc

    preprocessed_data = {
        'dataset': dataset
    }

    with open('../data/preprocessed_data.pickle', 'wb') as file:
        pickle.dump(preprocessed_data, file)