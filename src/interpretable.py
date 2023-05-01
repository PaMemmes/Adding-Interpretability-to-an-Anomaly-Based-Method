import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from collections import defaultdict
FILENAME = '../data/preprocessed_data.pickle'

from utils.utils import read_csv

def make_interpret_plots(explainer, shap_values, test_x, df_cols, name):
    f = shap.force_plot(explainer.expected_value, shap_values[:1000,:], test_x[:1000,:], feature_names=df_cols, show=False)
    shap.save_html(name + 'force_plot.htm', f)
    plt.close()
    print('After force plot')
    shap.summary_plot(shap_values, test_x, plot_type="bar", feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig(name + 'summary_bar.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('After first summary plot')
    shap.summary_plot(shap_values, test_x, feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig(name + 'summary.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('End interpreting...')

def seperate_dfs(df):
    dfs = defaultdict()
    for col in df['Label'].unique():
        print('Col', col)
        dfs[col] = df[df['Label']==col]
    
    len_dfs = [len(i) for i in dfs.values()]
    le = np.sum(len_dfs)
    print('Len dfs', le)
    print('len_dfs', len(df))
    return dfs

def interpret_tree(model, test, df_cols, save):
    name = '../experiments/' + save + '/best/'
    print('Starting interpreting...')

    df_cols = df_cols[:-1]
    test_df = pd.DataFrame(test.x, columns=df_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df)

    make_interpret_plots(explainer, shap_values, test.x, df_cols, name)

    # df = read_csv()
    # dfs = seperate_dfs(df)
    
    # for label, df in dfs.items():
    #     name = '../experiments/' + save + '/best/' + label + '_'
    #     df_cols = df.columns
    #     # df_cols = df_cols[:-1]
    #     explainer = shap.TreeExplainer(model)
    #     shap_values = explainer.shap_values(df)
    #     make_interpret_plots(explainer, shap_values, df, df_cols, name)

    