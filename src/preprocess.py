import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import glob

from utils.utils import DataSequence
from utils.utils import remove_infs, make_labels_binary, subset

BATCH_SIZE = 256

def preprocess(kind='normal', add_data=None):
    df = pd.read_csv('../data/cicids2017_kaggle/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
    df = df.rename(columns={' Label': 'label'})
    df_cols = list(df)
    print(df)
    print('Original length:', len(df))
    if add_data is not None:
        x = np.array(add_data)
        x = np.array(x).reshape((x.shape[0]*x.shape[1]), x.shape[2])
        y = np.array(['ANOMALY' for i in range(len(x))]).reshape(len(x), 1)
        data = np.concatenate((x,y), axis=1)
        df_add = pd.DataFrame(data, columns=df_cols)
        labels_add = df_add['label']
        x_train_add = df_add.drop('label',axis=1)
        le = LabelEncoder()
        le.fit(labels_add)
        int_labels_add = le.transform(labels_add)
        y_train_add = int_labels_add
    
    before_removal = len(df)
    df, labels = remove_infs(df)
    print(f'Length before NaN drop: {before_removal}, after NaN drop: {len(df)}\nThe df is now {len(df)/before_removal} of its original size')
    print('LABELS', labels)
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
    if kind == 'normal':
        x_train, y_train = subset(x_train, y_train, 0)
        print('Using only normal data')
    elif kind == 'anomaly':
        x_train, y_train = subset(x_train, y_train, 1)
        print('Using only anomaly data')

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if add_data is not None:
        x_train_add = x_train_add.to_numpy()
        print('x_train.shape', x_train.shape)
        print('x_train_add.shape', x_train_add.shape)
        print('y_train.shape', y_train.shape)
        print('y_train_add.shape', y_train_add.shape)
        x_train = np.vstack((x_train, x_train_add))
        y_train = np.concatenate((y_train, y_train_add))
        print('x_train.shape', x_train.shape)
        print('x_train_add.shape', x_train_add.shape)
        print('y_train.shape', y_train.shape)
        print('y_train_add.shape', y_train_add.shape)

        print('x_train', x_train)
        print('x_train_add', x_train_add)
        print('y_train_add', y_train_add)
        
    train_sqc = DataSequence(x_train, y_train, batch_size=BATCH_SIZE)
    test_sqc = DataSequence(x_test, y_test, batch_size=BATCH_SIZE)

    dataset = {}
    dataset['train'] = train_sqc
    dataset['test'] = test_sqc

    preprocessed_data = {
        'dataset': dataset,
        'cols': df_cols
    }

    with open('../data/preprocessed_data.pickle', 'wb') as file:
        pickle.dump(preprocessed_data, file)
    
    return train_sqc