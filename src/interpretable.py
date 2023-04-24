import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import pandas as pd
FILENAME = '../data/preprocessed_data.pickle'

def interpret_tree(model, train, test, df_cols, save):
    name = '../experiments/' + save + '/best/'
    print('Starting interpreting...')

    df_cols = df_cols[:-1]
    test_df = pd.DataFrame(test.x, columns=df_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df)

    f = shap.force_plot(explainer.expected_value, shap_values[:1000,:], test.x[:1000,:], feature_names=df_cols, show=False)
    shap.save_html(name + 'force_plot.htm', f)
    plt.close()
    print('After force plot')
    shap.summary_plot(shap_values, test.x, plot_type="bar", feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig(name + 'summary_bar.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('After first summary plot')
    shap.summary_plot(shap_values, test.x, feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig(name + 'summary.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('End interpreting...')