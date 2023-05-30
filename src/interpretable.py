import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

import pandas as pd
from collections import defaultdict

from preprocess import DataFrame

from utils.utils import read_csv, NumpyEncoder


def make_interpret_plots(explainer, shap_values, test_x, df_cols, name):
    if len(shap_values) > 1000:
        shap_values = shap_values[:1000]
        test_x = test_x[:1000]
    f = shap.force_plot(
        explainer.expected_value,
        shap_values,
        test_x,
        feature_names=df_cols,
        show=False)
    shap.save_html(name + 'force_plot.htm', f)
    plt.close()
    print(shap_values.shape)
    print(test_x.shape)
    shap.summary_plot(
        shap_values,
        test_x,
        plot_type="bar",
        feature_names=df_cols,
        show=False)
    f = plt.gcf()
    f.savefig(name + 'summary_bar.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    shap.summary_plot(shap_values, test_x, feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig(name + 'summary.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    for col in df_cols:
        print('Column', col)
        shap.dependence_plot(col, shap_values, test_x, show=False)
        f = plt.gcf()
        f.savefig(name + col.replace('/', '_')  + '_dependence.pdf', bbox_inches='tight', dpi=300)
        plt.close()

    shapleys = {
        'Expected_value': explainer.expected_value,
        'Shapleys': shap_values,
        'text_x': test_x.to_numpy()
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
    make_interpret_plots(
        explainer,
        shap_values,
        test_df,
        df_cols,
        name)

    test_df = pd.DataFrame(data.test_frag_sqc.x, columns=df_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df)
    make_interpret_plots(
        explainer,
        shap_values,
        test_df,
        df_cols,
        name_frags)

    data.seperate_dfs(filename=None)

    for label, test_sqc in data.seperate_tests.items():
        name = '../experiments/' + save + '/best/' + label + '/'
        Path(name).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(test_sqc.x, columns=df_cols)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        make_interpret_plots(explainer, shap_values, test_df, df_cols, name)
