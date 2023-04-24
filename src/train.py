from pathlib import Path
import json
import collections
import pickle
import shutil

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score

from hyperopt import hyperopt
from utils.utils import test_model, calc_metrics
from utils.wasserstein import HyperWGAN
from utils.gan import HyperGAN
from utils.plots import plot_confusion_matrix, plot_roc, plot_precision_recall

FILENAME = '../data/preprocessed_data.pickle'


def open_config(model_name):
    with open('configs/config_' + model_name + '.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
    return config

def get_preds(results, train):

    normals = collections.Counter(train.y)[0]
    anomalies = collections.Counter(train.y)[1]
    anomalies_percentage = anomalies / (normals + anomalies)
    
    # Obtaining the lowest "anomalies_percentage" score
    per = np.percentile(results, anomalies_percentage*100)
    results = np.array(results)
    y_pred = results.copy()
    probas = np.vstack((1-y_pred, y_pred)).T

    inds = y_pred > per
    inds_comp = y_pred <= per
    y_pred[inds] = 0
    y_pred[inds_comp] = 1
    return y_pred, probas, per, anomalies_percentage

def train(model_name, train, test, frags=None, num_trials=1, num_retraining=1, epochs=1, save=False):
    experiment = '../experiments/' + save + '/all/experiment'
    Path('../experiments/' + save + '/best/').mkdir(parents=True, exist_ok=True)

    config = open_config(model_name)

    best_hp = hyperopt(model_name, config, train, test, num_trials)

    num_features = train.x.shape[1]

    saves = []
    models = []
    for i in range(num_retraining):
        print('Starting experiment:', i)
        if save is not False:
            name = experiment + str(i) + '_tuner'
            Path(name).mkdir(parents=True, exist_ok=True)
            
        if model_name == 'wgan':
            hypermodel = HyperWGAN(num_features, config, discriminator_extra_steps=3, gp_weight=5.0)
        elif model_name == 'gan':
            hypermodel = HyperGAN(num_features, config)
        model = hypermodel.build(best_hp)

        hypermodel.fit(best_hp, model, train, epochs=epochs)

        results_df, results = test_model(hypermodel, test)
        models.append(hypermodel)
        y_pred, probas, per, anomalies_percentage = get_preds(results, train)

        precision, recall, f1, _ = precision_recall_fscore_support(test.y, y_pred, average='binary')
        accuracy = accuracy_score(test.y, y_pred)
        fpr, tpr, thresholds = roc_curve(test.y, probas[:,0])
        
        auc_val = auc(fpr, tpr)
        cm = confusion_matrix(test.y, y_pred)
        cm_norm = confusion_matrix(test.y, y_pred, normalize='all')
        mets = calc_metrics(cm)
        d = dict(mets)

        d_frag = dict()

        if save is not False:
            plot_confusion_matrix(cm, name + '/cm.pdf', save)
            plot_confusion_matrix(cm_norm, name + '/cm_normalized.pdf', save)
            plot_roc(tpr, fpr, auc_val, name + '/roc.pdf', save)
            plot_precision_recall(test.y, probas, name + '/precision_recall.pdf')
        results = {
                'Anomalies percentage': anomalies_percentage,
                'Cutoff': per,
                'Mean Score for normal packets': results_df.loc[results_df['y_test'] == 0, 'results'].mean(),
                'Mean Score for anomalous packets': results_df.loc[results_df['y_test'] == 1, 'results'].mean(),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc_val,
                'Best HP': best_hp['Dropout'],
                'Metrics': d,
                'I':i
        }

        if frags is not None:
            print(frags)
            results_df_frag, results_frag = test_model(model, frags)
            y_pred_frag, probas_frag, per_frag, anomalies_percentage_frag = get_preds(results_frag, train)

            precision_frag, recall_frag, f1_frag, _ = precision_recall_fscore_support(frags.y, y_pred_frag, average='binary')
            accuracy_frag = accuracy_score(frags.y, y_pred_frag)
            fpr_frag, tpr_frag, thresholds_frag = roc_curve(frags.y, probas_frag[:,0])
            auc_val_frag = auc(fpr_frag, tpr_frag)

            cm_frag = confusion_matrix(frags.y, y_pred_frag)
            cm_frag_norm = confusion_matrix(frags.y, y_pred_frag, normalize='all')
            mets_frag = calc_metrics(cm_frag)
            d_frag = dict(mets_frag)
            print('anomalies perc', anomalies_percentage_frag)
            print('Frags metrics: ', d_frag)
            if save is not False:
                plot_confusion_matrix(cm_frag, name + '/cm_frag.pdf', save)
                plot_confusion_matrix(cm_frag_norm, name + '/cm_frag_normalized.pdf', save)
                plot_roc(tpr, fpr, auc_val, name + '/frag_roc.pdf', save)
                plot_precision_recall(test.y, probas, name + '/frag_precision_recall.pdf')
                results = {
                        'Anomalies percentage': anomalies_percentage,
                        'Cutoff': per,
                        'Mean Score for normal packets': results_df.loc[results_df['y_test'] == 0, 'results'].mean(),
                        'Mean Score for anomalous packets': results_df.loc[results_df['y_test'] == 1, 'results'].mean(),
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1,
                        'AUC': auc_val,
                        'Best HP': best_hp['Dropout'],
                        'Metrics': d,
                        'Accuracy Frag': accuracy_frag,
                        'Precision Frag': precision_frag,
                        'Recall Frag': recall_frag,
                        'F1 Frag': f1_frag,
                        'AUC Frag': auc_val_frag,
                        'Metrics Frag': d_frag,
                        'I':i
                }

        if save is not False:
            with open(name + '/' + save + '.json', 'w', encoding='utf-8') as f: 
                json.dump(results, f, ensure_ascii=False, indent=4)
        saves.append(results)
    
    best_res = sorted(saves, key=lambda d: d['Accuracy'])[-1]
    print('Best result: ', best_res)
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/cm.pdf', '../experiments/' + save + '/best/cm.pdf')
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/cm_normalized.pdf', '../experiments/' + save + '/best/cm_normalized.pdf')
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/roc.pdf', '../experiments/' + save + '/best/roc.pdf')
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/precision_recall.pdf', '../experiments/' + save + '/best/precision_recall.pdf')
    
    if frags is not None:
        shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/cm_frag.pdf', '../experiments/' + save + '/best/cm_frag.pdf')
        shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/cm_frag_normalized.pdf', '../experiments/' + save + '/best/cm_frag_normalized.pdf')
        shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/frag_roc.pdf', '../experiments/' + save + '/best/frag_roc.pdf')
        shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/frag_precision_recall.pdf', '../experiments/' + save + '/best/frag_precision_recall.pdf')


    if save is not False:
        with open('../experiments/' + save + '/best/best_model_wgan.json', 'w', encoding='utf-8') as f: 
            json.dump(best_res, f, ensure_ascii=False, indent=4)
    return models[best_res['I']]
