import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import glob

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


if __name__ =='__main__':
    df = pd.read_csv('data/cicids2017_kaggle/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
    df = df.rename(columns={' Label': 'Label'})

    before_removal = len(df)
    df, labels = remove_infs(df)
    print(f'Length before NaN drop: {before_removal}, after NaN drop: {len(df)}\n \
    The df is now {len(df)/before_removal} of its original size')

    le = LabelEncoder()
    le.fit(labels)

    int_labels = le.transform(labels)
    df = df.drop('Label', axis=1)


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

    with open('data/preprocessed_data.pickle', 'wb') as file:
        pickle.dump(preprocessed_data, file)