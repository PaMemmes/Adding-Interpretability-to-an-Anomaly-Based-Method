import pickle
import collections

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os
from utils.utils import DataSequence
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from utils.utils import remove_infs, make_labels_binary, subset, encode
from dataclasses import dataclass

from typing import Any
from collections import defaultdict
BATCH_SIZE = 256

@dataclass
class DataFrame:
    filename: str = None

    df: pd.DataFrame() = None
    df_frag: pd.DataFrame() = None
    df_add: pd.DataFrame() = None

    df_cols: object = None

    le: LabelEncoder() = None

    x_test: object = None
    y_test: object = None

    x_test_frags: object = None
    y_test_frags: object = None

    train_sqc: Any = None
    test_sqc: Any = None

    test_frag_sqc: Any = None

    dfs: defaultdict() = None
    seperate_tests: defaultdict() = None

    def create_df(self, filename):
        if filename is not None:
            self.df = pd.read_csv('/mnt/md0/files_memmesheimer/cicids2018/' + filename)
        else:
            all_files = glob.glob(os.path.join('../data/cicids2018', "*.csv"))
            self.df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

        self.df = self.df.sample(frac=1)
        self.df = self.df[:50000]
        self.df = self.df.drop('Timestamp', axis=1)
        self.df_cols = self.df.columns

    def create_label_encoder(self):
        # Auxilliary df for LabelEncoder() to encode the right number of labels
        df_exe = pd.read_csv('/mnt/md0/files_memmesheimer/csv_fragmentedV3/All.ElectroRAT.pcap_Flow.csv')
        df_exe = df_exe.drop(['Dst IP', 'Flow ID', 'Src IP', 'Src Port', 'Timestamp'], axis=1)
        df_exe['Label'] = 'Fragmented Malware'
        df_exe.loc[1, 'Label'] = 'ANOMALY'
        df = pd.concat([df_exe, self.df], ignore_index=True)
        labels = df['Label']
        self.le = LabelEncoder()
        self.le.fit(labels)

    def create_oos_test(self, test_size):
        # Make-out-of-sample test split, s.t. additional data is not incorporated
        df, labels = remove_infs(self.df)
        labels = encode(self.le, labels)
        labels = make_labels_binary(self.le, labels)
        _, self.x_test, y_train, self.y_test = train_test_split(df, labels, test_size=test_size, shuffle=False)
        self.df = self.df[:int((1-test_size)*len(self.df))]
        
        normals = collections.Counter(y_train)[0]
        anomalies = collections.Counter(y_train)[1]
        self.anomalies_percentage = anomalies / (normals + anomalies)

    def make_frags(self, test_size):
        all_files = glob.glob(os.path.join('/mnt/md0/files_memmesheimer/csv_fragmentedV3/', "*.csv"))
        self.df_frag = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        self.df_frag = self.df_frag.drop(['Dst IP', 'Flow ID', 'Src IP', 'Src Port', 'Timestamp'], axis=1)
        assert len(self.df_frag.columns), len(self.df_cols)

        self.df_frag['Label'] = 'Fragmented Malware'
        self.df_frag['Dst Port'] = np.random.randint(0, self.df.max(axis=0)['Dst Port'], size=(len(self.df_frag)))
        
        df = self.df_frag
        df, labels = remove_infs(df)

        labels = encode(self.le, labels)
        labels = make_labels_binary(self.le, labels)

        _, self.x_test_frags, _, self.y_test_frags = train_test_split(df, labels, test_size=test_size, shuffle=False)
        
        print('Length of frags test: ', len(self.x_test_frags))
        print('Length of normal test_data', len(self.x_test))

    def preprocess_add(self, add_data):
        x = np.array(add_data)
        x = np.array(x).reshape((x.shape[0]*x.shape[1]), x.shape[2])
        y = np.array(['ANOMALY' for i in range(len(x))]).reshape(len(x), 1)
        data = np.concatenate((x,y), axis=1)
        self.df_add = pd.DataFrame(data, columns=self.df_cols)
    
    def preprocess(self, filename = None, kind=None, frags=False, add=None, test_size=0.15):
        self.create_df(filename)
        self.create_label_encoder()
        self.create_oos_test(test_size)
        self.make_frags(test_size=test_size)
        if frags is True:
            self.df = pd.concat([self.df_frag.iloc[int((1-test_size)*len(self.df_frag)):], self.df], ignore_index=True)
        if add is not None:
            self.df = pd.concat([self.df_add, self.df], ignore_index=True)
            self.df = self.df.sample(frac=1)
        df, labels = remove_infs(self.df)
        labels = encode(self.le, labels)
        labels = make_labels_binary(self.le, labels)
        x_train, _, y_train, _ = train_test_split(df, labels, test_size=test_size, shuffle=False)

        # Subsetting only Normal Network packets in training set
        if kind == 'normal':
            x_train, y_train = subset(x_train, y_train, 0)
        elif kind == 'anomaly':
            x_train, y_train = subset(x_train, y_train, 1)
            print('Using only anomaly data')

        scaler = MinMaxScaler()

        x_train = scaler.fit_transform(x_train)
        self.x_test = scaler.transform(self.x_test)
        self.x_test_frags = scaler.transform(self.x_test_frags)
        
        self.train_sqc = DataSequence(x_train, y_train, batch_size=BATCH_SIZE)
        self.test_sqc = DataSequence(self.x_test, self.y_test, batch_size=BATCH_SIZE)

        self.test_frag_sqc = DataSequence(self.x_test_frags, self.y_test_frags, batch_size=BATCH_SIZE)
        
    def create_df_only_normal(self, filename):
        if filename is not None:
            df_all = pd.read_csv('/mnt/md0/files_memmesheimer/cicids2018/' + filename)
        else:
            all_files = glob.glob(os.path.join('/mnt/md0/files_memmesheimer/cicids2018', "*.csv"))
            df_all = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        dfs = defaultdict()
        self.seperate_tests = defaultdict()
        for col in df_all['Label'].unique():
            dfs[col] = df_all[df_all['Label'] == col]
        self.df = dfs['Benign']
        
        self.df = self.df[:50000]
        self.df = self.df.drop('Timestamp', axis=1)
        self.df_cols = self.df.columns
    
        
    def seperate_dfs(self, filename, test_size=0.15):
        if filename is not None:
            df_all = pd.read_csv('/mnt/md0/files_memmesheimer/cicids2018/' + filename)
        else:
            all_files = glob.glob(os.path.join('/mnt/md0/files_memmesheimer/cicids2018', "*.csv"))
            df_all = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

        _df = df_all.copy()
        _df = _df.drop('Timestamp', axis=1)
        _df, _labels = remove_infs(_df)
        _, _x_test, _, _ = train_test_split(_df, _labels, test_size=1, shuffle=False)
        scaler = MinMaxScaler()
        scaler.fit(_x_test)

        dfs = defaultdict()
        self.seperate_tests = defaultdict()
        for col in df_all['Label'].unique():
            dfs[col] = df_all[df_all['Label'] == col]
            df = dfs[col].sample(frac=1)
            df = df[:50000]
            df = df.drop('Timestamp', axis=1)
            df, labels = remove_infs(df)
            labels = encode(self.le, labels)
            x_train, x_test, _, y_test = train_test_split(df, labels, test_size=1, shuffle=False)
            
            self.x_test = scaler.transform(self.x_test)

            test_sqc = DataSequence(self.x_test, y_test, batch_size=BATCH_SIZE)
            self.seperate_tests[col] = test_sqc

    # def preprocess_anomalies_only_frags(self, filename = None, kind=None, frags=False, add=None, test_size=0.15):
    #     self.create_df_only_normal(filename)
    #     self.create_label_encoder()
    #     self.create_oos_test(test_size)
    #     self.make_frags(test_size)
    #     print('LEN DF', len(self.df))
    #     self.df = pd.concat([self.df_frag.iloc[int((1-test_size)*len(self.df_frag)):], self.df], ignore_index=True)
        
    #     print('LEN DF after concat', len(self.df))
    #     if add is not None:
    #         self.df = pd.concat([self.df_add, self.df], ignore_index=True)
    #     self.df = self.df.sample(frac=1)
    #     df, labels = remove_infs(self.df)
    #     labels = encode(self.le, labels)
    #     labels = make_labels_binary(self.le, labels)
    #     x_train, _, y_train, _ = train_test_split(df, labels, test_size=test_size, shuffle=False)

    #     # Subsetting only Normal Network packets in training set
    #     if kind == 'normal':
    #         x_train, y_train = subset(x_train, y_train, 0)
    #     elif kind == 'anomaly':
    #         x_train, y_train = subset(x_train, y_train, 1)
    #         print('Using only anomaly data')

    #     scaler = MinMaxScaler()

    #     x_train = scaler.fit_transform(x_train)
    #     self.x_test = scaler.transform(self.x_test)

    #     self.x_test_frags = scaler.transform(self.x_test_frags)
        
    #     self.train_sqc = DataSequence(x_train, y_train, batch_size=BATCH_SIZE)
    #     self.test_sqc = DataSequence(self.x_test, self.y_test, batch_size=BATCH_SIZE)
        
    #     self.test_frag_sqc = DataSequence(self.x_test_frags, self.y_test_frags, batch_size=BATCH_SIZE)
