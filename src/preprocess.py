import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import glob

from utils.utils import remove_infs, make_labels_binary

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

    x_train, x_test, y_train, y_test = train_test_split(df,
                                                    int_labels,
                                                    test_size=.15)
    
    preprocessed_data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'le': le
    }

    with open('../data/preprocessed_data.pickle', 'wb') as file:
        pickle.dump(preprocessed_data, file)