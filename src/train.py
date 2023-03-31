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

def get_preds(results, test):

    normals = collections.Counter(test.y)[0]
    anomalies = collections.Counter(test.y)[1]
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

def train(model_name, num_trials, num_retraining, epochs, save=False):
    experiment = '../experiments/' + save + '/all/experiment'

    input_file = open(FILENAME, 'rb')
    preprocessed_data = pickle.load(input_file)
    input_file.close()

    config = open_config(model_name)

    dataset = preprocessed_data['dataset']
    train = dataset['train']
    test = dataset['test']
    best_hp = hyperopt(model_name, config, train, test, num_trials)

    num_features = train.x.shape[1]

    saves = []
    models = []
    for i in range(num_retraining):
        print('Starting experiment:', i)
        name = experiment + str(i) + '_tuner'
        if save is not False:
            Path(name).mkdir(parents=True, exist_ok=True)
        if model_name == 'wgan':
            hypermodel = HyperWGAN(num_features, config, discriminator_extra_steps=3, gp_weight=5.0)
        elif model_name == 'gan':
            hypermodel = HyperGAN(num_features, config)
        model = hypermodel.build(best_hp)

        hypermodel.fit(best_hp, model, train, epochs=epochs)

        results_df, results = test_model(hypermodel, test)
        models.append(hypermodel)
        y_pred, probas, per, anomalies_percentage = get_preds(results, test)

        precision, recall, f1, _ = precision_recall_fscore_support(test.y, y_pred, average='binary')
        accuracy = accuracy_score(test.y, y_pred)
        fpr, tpr, thresholds = roc_curve(test.y, probas[:,0])
        
        auc_val = auc(fpr, tpr)
        cm = confusion_matrix(test.y, y_pred)
        mets = calc_metrics(cm)
        d = dict(mets)
        if save is not False:
            plot_confusion_matrix(cm, name + '/confusion.png', save)
            plot_roc(tpr, fpr, auc_val, name + '/roc.png', save)
            plot_precision_recall(test.y, probas, name + '/precision_recall.png')
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
        if save is not False:
            with open(name + '/' + save + '.json', 'w', encoding='utf-8') as f: 
                json.dump(results, f, ensure_ascii=False, indent=4)
        saves.append(results)
    
    best_res = sorted(saves, key=lambda d: d['Accuracy'])[-1]
    print('Best result: ', best_res)
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/confusion.png', '../experiments/' + save + '/best/confusion.png')
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/roc.png', '../experiments/' + save + '/best/roc.png')
    shutil.copy(experiment + str(best_res['I']) + '_tuner' + '/precision_recall.png', '../experiments/' + save + '/best/precision_recall.png')
   
    if save is not False:
        with open('../experiments/' + save + '/best/best_model.json', 'w', encoding='utf-8') as f: 
            json.dump(best_res, f, ensure_ascii=False, indent=4)
    return models[best_res['I']]
