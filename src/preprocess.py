import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os
from utils.utils import DataSequence
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from utils.utils import remove_infs, make_labels_binary, subset, scale, encode

BATCH_SIZE = 256

class DataFrame():
    def __init__(self):
        self.df = pd.read_csv('../data/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
        self.df = self.df[:50000]
        self.df = self.df.drop('Timestamp', axis=1)
        self.df_cols = self.df.columns

        # Auxialliary df for LabelEncoder() to encode the right number of labels
        df_exe = pd.read_csv('../data/csv_fragmentedV3/All.ElectroRAT.pcap_Flow.csv')
        df_exe = df_exe.drop(['Dst IP', 'Flow ID', 'Src IP', 'Src Port', 'Timestamp'], axis=1)
        df_exe['Label'] = 'Fragmented Malware'
        df_exe.loc[1, 'Label'] = 'ANOMALY'
        df = pd.concat([df_exe, self.df], ignore_index=True)
        labels = df['Label']
        self.le = LabelEncoder()
        self.le.fit(labels)

        # Make-out-of-sample test split, s.t. additional data is not incorporated
        df, labels = remove_infs(self.df)
        labels = encode(self.le, labels)
        labels = make_labels_binary(self.le, labels)
        _, self.x_test, _, self.y_test = train_test_split(df, labels, test_size=0.2)

    def preprocess_frag(self):
        all_files = glob.glob(os.path.join('../data/csv_fragmentedV3', "*.csv"))
        self.df_frag = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        self.df_frag = self.df_frag.drop(['Dst IP', 'Flow ID', 'Src IP', 'Src Port', 'Timestamp'], axis=1)
        assert len(self.df_frag.columns), len(self.df_cols)

        self.df_frag['Label'] = 'Fragmented Malware'
        self.df_frag['Dst Port'] = np.random.randint(0, self.df.max(axis=0)['Dst Port'],size=(len(self.df_frag)))

    def make_frag_main_df(self):
        df = self.df_frag
        df, labels = remove_infs(df)

        labels = encode(self.le, labels)
        labels = make_labels_binary(self.le, labels)

        x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2)

        x_train, x_test = scale(x_train, x_test)

        self.train_sqc = DataSequence(x_train, y_train, batch_size=BATCH_SIZE)
        self.test_sqc = DataSequence(x_test, y_test, batch_size=BATCH_SIZE)

        return self.train_sqc, self.test_sqc, self.df_cols

    def preprocess_add(self, gan_data):
        x = np.array(gan_data)
        x = np.array(x).reshape((x.shape[0]*x.shape[1]), x.shape[2])
        y = np.array(['ANOMALY' for i in range(len(x))]).reshape(len(x), 1)
        data = np.concatenate((x,y), axis=1)
        self.df_add = pd.DataFrame(data, columns=self.df_cols)
    
    def merge_df_add(self):
        self.df = pd.concat([self.df, self.df_add], ignore_index=True)
    
    def merge_df_frag(self):
        self.df = pd.concat([self.df, self.df_frag], ignore_index=True)

    def merge_df_all(self):
        self.df = pd.concat([self.df, self.df_add], ignore_index=True)
        self.df = pd.concat([self.df, self.df_frag], ignore_index=True)

    def prepare(self, kind=None):
        self.df = self.df.sample(frac=1)
        df, labels = remove_infs(self.df)
        labels = encode(self.le, labels)
        labels = make_labels_binary(self.le, labels)
        x_train, _, y_train, _ = train_test_split(df, labels, test_size=0.2)

        # Subsetting only Normal Network packets in training set
        if kind == 'normal':
            x_train, y_train = subset(x_train, y_train, 0)
        elif kind == 'anomaly':
            x_train, y_train = subset(x_train, y_train, 1)
            print('Using only anomaly data')

        x_train, self.x_test = scale(x_train, self.x_test)

        self.train_sqc = DataSequence(x_train, y_train, batch_size=BATCH_SIZE)
        self.test_sqc = DataSequence(self.x_test, self.y_test, batch_size=BATCH_SIZE)


def get_frags():
    data = DataFrame()
    data.preprocess_frag()
    train, test, df_cols = data.make_frag_main_df()
    return train, test, df_cols

def preprocess(kind=None, additional=None, frag_data=False):
    data = DataFrame()
    
    if kind is None and frag_data == False and additional is None:
        data.prepare()
        return data
    elif kind is None and frag_data == True and additional is not None:
        data.preprocess_add(additional)
        data.preprocess_frag()
        data.merge_df_all()
        data.prepare()
        return data
    elif kind is None and frag_data == True and additional is None:
        data.preprocess_frag()
        data.merge_df_frag()
        data.prepare()
        return data
    elif kind is None and frag_data == False and additional is not None:
        data.preprocess_add(additional)
        data.merge_df_add()
        data.prepare()
        return data
    else:
        data.prepare(kind)
        return data
