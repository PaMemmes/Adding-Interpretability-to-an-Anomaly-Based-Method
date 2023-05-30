import pandas as pd
import os
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from utils.utils import subset

plt.style.use(['ieee', 'science'])

def bar_plot_agg(df):
    
    df['Label'] = df['Label'].replace({'DoS-attacks-SlowHTTPTest': 'DoS', 'DoS attacks-Slowloris': 'DoS', 'DoS attacks-Hulk': 'DoS'}, regex=True)
    df['Label'] = df['Label'].replace({'DDoS attack-LOIC-UDP': 'DDoS', 'DDoS attack-HOIC': 'DDoS', 'DDoS attacks-LOIC-HTTP': 'DDoS'}, regex=True)
    df['Label'] = df['Label'].replace({'Brute Force -Web': 'Brute Force', 'Brute Force -XSS': 'Brute Force', 'SSH-BruteForce': 'Brute Force', 'FTP:BruteForce':'Brute Force'}, regex=True)

    label_counts = df['Label'].value_counts(normalize=True) * 100
    label_counts = label_counts.sort_values(ascending=False)
 
    sns.set_style("whitegrid", {'axes.grid' : True})
    ax = sns.barplot(x=label_counts.index.tolist(), y = label_counts.to_numpy())
    ax.set_xticklabels(ax.get_xticklabels())
    plt.tight_layout()
    ax.set(xlabel = 'Attack Type', ylabel='Percentage')
    ax.bar_label(ax.containers[0], fmt='%.3f')
    fig = ax.get_figure()
    fig.savefig("../distribution_cicids2018_agg.pdf", bbox_inches="tight")
    plt.close()

def bar_plot_binary(df):
    
    subset_benign = df[df['Label'] == 'Benign']
    subset_anomaly = df[df['Label'] != 'Benign']

    total_count = len(df)
    percentage_benign = len(subset_benign) / total_count * 100
    percentage_anomaly = len(subset_anomaly) / total_count * 100

    sns.set_style("darkgrid", {'axes.grid' : True})

    ax = sns.barplot(x=['Benign', 'Anomaly'], y = [percentage_benign, percentage_anomaly])
    ax.set_xticklabels(ax.get_xticklabels())
    plt.tight_layout()
    ax.set(xlabel = 'Attack Type', ylabel='Percentage')
    ax.bar_label(ax.containers[0], fmt='%.3f')
    fig = ax.get_figure()
    fig.savefig("../distribution_cicids2018_binary.pdf", bbox_inches="tight")
    plt.close()


def bar_plot(df):

    label_counts = df['Label'].value_counts(normalize=True) * 100
    label_counts = label_counts.sort_values(ascending=False)
    sns.set_style("whitegrid", {'axes.grid': True})
    ax = sns.barplot(x = label_counts.index.tolist(), y=label_counts)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    ax.set(xlabel = 'Attack Type', ylabel='Percentage')
    ax.bar_label(ax.containers[0], fmt='%.3f')
    fig = ax.get_figure()
    fig.savefig("../distribution_cicids2018.pdf", bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    # if not df:
    #     all_files = glob.glob(
    #     os.path.join(
    #         '../data/cicids2018',
    #         "*.csv"))
    #     df = pd.concat((pd.read_csv(f, engine='python') for f in all_files), ignore_index=True)

    df = pd.read_csv(
        '../data/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv')
    df = df[:50000]
    pd.set_option('display.max_columns', 500)

    bar_plot(df)
    bar_plot_binary(df)
    bar_plot_agg(df)
