from pathlib import Path
import json
import pickle
import shutil

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score

from hyperopt import hyperopt
from utils.utils import test_model, calc_metrics, calc_all_nn, get_preds, open_config
from utils.wasserstein import HyperWGAN
from utils.gan import HyperGAN
from utils.plots import plot_confusion_matrix, plot_roc, plot_precision_recall

FILENAME = '../data/preprocessed_data.pickle'


def train(model_name, train, test, frags=None, trials=1, num_retraining=1, epochs=1, save=False):
    experiment = '../experiments/' + save + '/all/experiment'
    Path('../experiments/' + save + '/best/').mkdir(parents=True, exist_ok=True)

    config = open_config(model_name)

    best_hp = hyperopt(model_name, config, train, test, trials)

    num_features = train.x.shape[1]

    saves = []
    models = []
    for i in range(num_retraining):
        print('Starting experiment:', i)
        name = experiment + str(i) + '_tuner'
        Path(name).mkdir(parents=True, exist_ok=True)
        
        if model_name == 'wgan':
            hypermodel = HyperWGAN(num_features, config, discriminator_extra_steps=3, gp_weight=5.0)
        elif model_name == 'gan':
            hypermodel = HyperGAN(num_features, config)
        model = hypermodel.build(best_hp)

        hypermodel.fit(best_hp, model, train, epochs=epochs)
        models.append(hypermodel)

        results_df, results = test_model(hypermodel, test)
        y_pred, probas, per, anomalies_percentage = get_preds(results, train)
        metrics, cm, cm_norm = calc_all_nn(test, y_pred, probas)
        plot_confusion_matrix(cm, name + '/cm.pdf', save)
        plot_confusion_matrix(cm_norm, name + '/cm_normalized.pdf', save)
        plot_roc(metrics['TPR'], metrics['FPR'], metrics['AUC'], name + '/roc.pdf', save)
        #plot_precision_recall(test.y, probas, name + '/precision_recall.pdf')
        
        results = {
                'Anomalies percentage': anomalies_percentage,
                'Cutoff': per,
                'Accuracy': metrics['ACC'],
                'Mean Score for normal packets': results_df.loc[results_df['y_test'] == 0, 'results'].mean(),
                'Mean Score for anomalous packets': results_df.loc[results_df['y_test'] == 1, 'results'].mean(),
                'Best HP': best_hp['Dropout'],
                'Metrics test': metrics,
                'I':i
        }

        if frags is not None:
            results_df_frag, results_frag = test_model(model, frags)
            y_pred_frag, probas_frag, per_frag, anomalies_percentage_frag = get_preds(results_frag, train)
            
            metrics_frag, cm_frag, cm_norm_frag = calc_all_nn(frags, y_pred_frag, probas_frag)

            plot_confusion_matrix(cm_frag, name + '/cm_frag.pdf', save)
            plot_confusion_matrix(cm_norm_frag, name + '/cm_frag_normalized.pdf', save)
            plot_roc(metrics_frag['TPR'], metrics_frag['FPR'], metrics_frag['AUC'], name + '/frag_roc.pdf', save)
            #plot_precision_recall(frags.y, probas, name + '/frag_precision_recall.pdf')
            
            results = {
                    'Anomalies percentage': anomalies_percentage,
                    'Cutoff': per,
                    'Accuracy': metrics['ACC'],
                    'Accuracy Frag': metrics_frag['ACC'],
                    'Mean Score for normal packets': results_df.loc[results_df['y_test'] == 0, 'results'].mean(),
                    'Mean Score for anomalous packets': results_df.loc[results_df['y_test'] == 1, 'results'].mean(),
                    'Best HP': best_hp['Dropout'],
                    'Metrics test': metrics,
                    'Metrics frag': metrics_frag,
                    'I':i
            }
        with open(name + '/' + save + '.json', 'w', encoding='utf-8') as f: 
            json.dump(results, f, ensure_ascii=False, indent=4)
        saves.append(results)
    
    best_res = sorted(saves, key=lambda d: d['Accuracy'])[-1]
    print('Best result: ', best_res)
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/cm.pdf', '../experiments/' + save + '/best/cm.pdf')
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/cm_normalized.pdf', '../experiments/' + save + '/best/cm_normalized.pdf')
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/roc.pdf', '../experiments/' + save + '/best/roc.pdf')
    # shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/precision_recall.pdf', '../experiments/' + save + '/best/precision_recall.pdf')
    
    if frags is not None:
        shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/cm_frag.pdf', '../experiments/' + save + '/best/cm_frag.pdf')
        shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/cm_frag_normalized.pdf', '../experiments/' + save + '/best/cm_frag_normalized.pdf')
        shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/frag_roc.pdf', '../experiments/' + save + '/best/frag_roc.pdf')
        # shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/frag_precision_recall.pdf', '../experiments/' + save + '/best/frag_precision_recall.pdf')


    if save is not False:
        with open('../experiments/' + save + '/best/best_model_wgan.json', 'w', encoding='utf-8') as f: 
            json.dump(best_res, f, ensure_ascii=False, indent=4)
    return models[best_res['I']]
