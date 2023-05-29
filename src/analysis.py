import pandas as pd
import os
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# def bar_plot_old():
#     y = [83.070, 7.786, 4.031, 2.347, 1.763, 0.997, 0.006]
#     x = ['Benign', 'DDoS', 'DoS', 'Brute Force', 'Botnet', 'Infiltration', 'Web Attack']
#     data = pd.DataFrame({"Percentage": y, "Attack Type": x})
#     sns.set_style("whitegrid", {'axes.grid' : True})
#     ax = sns.barplot(x=data['Attack Type'], y = data['Percentage'])
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
#     plt.tight_layout()
#     ax.bar_label(ax.containers[0], fmt='%.3f')
#     fig = ax.get_figure()
#     fig.savefig("../distribution_cicids2018.pdf", bbox_inches="tight")
#     plt.close()

def bar_plot_real(df):
    all_files = glob.glob(
    os.path.join(
        '../data/cicids2018',
        "*.csv"))
    df = pd.concat((pd.read_csv(f, engine='python') for f in all_files), ignore_index=True)

    sns.set_style("whitegrid", {'axes.grid': True})
    ax = sns.histplot(data=df, x='Label', stat='probability')
    plt.tight_layout()
    ax.set(ylabel='Percent')
    fig = ax.get_figure()
    fig.savefig("../distribution_cicids2018.pdf", bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    df = pd.read_csv(
        '../data/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
    df = df[:50000]
    pd.set_option('display.max_columns', 500)
    print(df.head())
    print(len(df.columns))

    bar_plot_real(df)
