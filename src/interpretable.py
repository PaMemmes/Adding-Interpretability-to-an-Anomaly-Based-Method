import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import scienceplots

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

    shap.summary_plot(
        shap_values,
        test_x,
        plot_type="bar",
        feature_names=df_cols,
        show=False)
    f = plt.gcf()
    f.savefig(name + 'summary_bar.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    shap.summary_plot(
            shap_values,
            test_x,
            plot_type="bar",
            feature_names=df_cols,
            max_display=len(df_cols),
            show=False
            )
    f = plt.gcf()
    f.savefig(name + 'summary_bar_all.pdf', bbox_inches='tight', dpi=300)
    plt.close()


    shap.summary_plot(shap_values, test_x, feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig(name + 'summary.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    shap.summary_plot(shap_values, test_x, feature_names=df_cols, max_display=len(df_cols), show=False)
    f = plt.gcf()
    f.savefig(name + 'summary_all.pdf', bbox_inches='tight', dpi=300)
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

def feature_importance(model, df_cols, importance_type):
    weight = model.get_booster().get_score(importance_type=importance_type)
    print(weight)
    print('WEIGGHT MY NIGAG', len(weight))
    print('\n\n\n\n\n\n\n')
    sorted_idx = np.argsort(list(weight.values()))
    weight = np.sort(list(weight.values()))
    y = df_cols[sorted_idx]
    return weight, y

def plot_importance(model, name, df_cols, importance_type):
    plt.style.use(['ieee', 'science'])
    plt.style.use('seaborn-colorblind')
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    weight, y = feature_importance(model, df_cols, importance_type)
    if (len(weight) > 25):
        weight = weight[:25]
        y = y[:25]
    ax.barh(y=y, width=weight)
    ax.set_xlabel(importance_type.capitalize() + ' Score')
    ax.set_ylabel('Feature')
    fig.savefig(name + importance_type + '_importance.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def interpret_tree(model, data, save):
    name = '../experiments/' + save + '/best/'
    name_frags = '../experiments/' + save + '/best/frags'

    print('Starting interpreting...')

    df_cols = data.df_cols[:-1]
    test_df = pd.DataFrame(data.test_sqc.x, columns=df_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df, check_additivity=False)
    make_interpret_plots(
        explainer,
        shap_values,
        test_df,
        df_cols,
        name)
    
    plot_importance(model, name, df_cols, 'gain')
    plot_importance(model, name, df_cols, 'weight')

    test_df = pd.DataFrame(data.test_frag_sqc.x, columns=df_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df, check_additivity=False)
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
        shap_values = explainer.shap_values(df, check_additivity=False)
        make_interpret_plots(explainer, shap_values, df, df_cols, name)
