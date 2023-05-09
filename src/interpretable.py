import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json

import pandas as pd
from collections import defaultdict

from preprocess import DataFrame

FILENAME = 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'

from utils.utils import read_csv, NumpyEncoder

def make_interpret_plots(explainer, shap_values, test_x, df_cols, name):
    if len(shap_values) > 1000:
        shap_values = shap_values[:1000, :]
        test_x = test_x[:1000,:]
    f = shap.force_plot(explainer.expected_value, shap_values, test_x, feature_names=df_cols, show=False)
    shap.save_html(name + 'force_plot.htm', f)
    plt.close()

    shap.summary_plot(shap_values, test_x, plot_type="bar", feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig(name + 'summary_bar.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    shap.summary_plot(shap_values, test_x, feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig(name + 'summary.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    shapleys = {
            'Expected_value': explainer.expected_value,
            'Shapleys': shap_values,
            'text_x': test_x

    }
    dumped = json.dumps(shapleys, cls=NumpyEncoder)
    with open(name + 'shapley.json', 'w', encoding='utf-8') as f: 
        json.dump(dumped, f)

def interpret_tree(model, data, save):
    name = '../experiments/' + save + '/best/'
    name_frags = '../experiments/' + save + '/best/frags'
    print('Starting interpreting...')

    df_cols = data.df_cols[:-1]
    test_df = pd.DataFrame(data.test_sqc.x, columns=df_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df)
    make_interpret_plots(explainer, shap_values, data.test_sqc.x, df_cols, name)

    test_df = pd.DataFrame(data.test_frag_sqc.x, columns=df_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df)
    make_interpret_plots(explainer, shap_values, data.test_frag_sqc.x, df_cols, name_frags)


    data.seperate_dfs(FILENAME)

    for label, test_sqc in data.seperate_tests.items():
        name = '../experiments/' + save + '/best/' + label + '_'
        
        df = pd.DataFrame(test_sqc.x, columns=df_cols)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        make_interpret_plots(explainer, shap_values, test_sqc.x, df_cols, name)

    