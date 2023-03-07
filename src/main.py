import pickle
from pathlib import Path

import json
import collections
import pickle

from hyperopt import hyperopt
from utils.utils import test_model
from utils.network import HyperGAN
from utils.plots import plot_confusion_matrix, plot_roc, plot_losses, plot_precision_recall
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score

import tensorflow as tf
import numpy as np
import pandas as pd

if __name__ == '__main__':

    filename = '../data/preprocessed_data.pickle'

    input_file = open(filename, 'rb')
    preprocessed_data = pickle.load(input_file)
    input_file.close()

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())


    dataset = preprocessed_data['dataset']
    num_features = dataset['train'].x.shape[1]
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']

    best_hp = hyperopt()
    print("BEST HP: ", best_hp['Dropout'])
    #print("BEST HP: ", best_hp['activation function'])
    for i in range(20):
        name = '../experiments/experiment' + str(i) + '_tuner'
        Path(name).mkdir(parents=True, exist_ok=True)
        hypermodel = HyperGAN(num_features, config)
        model = hypermodel.build(best_hp)

        hypermodel.fit(best_hp, model, train)

        results_df, results = test_model(hypermodel, test)
        score_normal = results_df.loc[results_df['y_test'] == 0, 'results'].mean()
        score_anomalous = results_df.loc[results_df['y_test'] == 1, 'results'].mean()

        print('Mean score for normal packets :', score_normal)
        print('Mean score for anomalous packets :', score_anomalous)
        normals = collections.Counter(test.y)[0]
        anomalies = collections.Counter(test.y)[1]
        anomalies_percentage = anomalies / (normals + anomalies)

        # Obtaining the lowest "anomalies_percentage" score
        per = np.percentile(results, anomalies_percentage*100)
        y_pred = results.copy()
        y_pred = np.array(y_pred)
        probas = np.vstack((1-y_pred, y_pred)).T
        
        plot_precision_recall(test.y, probas, name + '/precision_recall_only_cic.png')
        
        # Thresholding based on the score
        inds = y_pred > per
        inds_comp = y_pred <= per
        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(
            test.y, y_pred, average='binary')
        accuracy = accuracy_score(test.y, y_pred)
        print('Accuracy Score :', accuracy)
        print('Precision :', precision)
        print('Recall :', recall)
        print('F1 :', f1)

        fpr, tpr, thresholds = roc_curve(test.y, y_pred)
        auc_curve = auc(fpr, tpr)
        plot_roc(tpr, fpr, auc_curve, name + '/roc_gan_only_cic.png', 'GAN')
        cm = confusion_matrix(test.y, y_pred)
        plot_confusion_matrix(cm, name + '/confusion_gan_only_cic.png', 'GAN')
        
        results = {
                'Normals (%)': 1 - anomalies_percentage,
                'Anomalies (%)': anomalies_percentage,
                'Mean Score for normal packets': results_df.loc[results_df['y_test'] == 0, 'results'].mean(),
                'Mean Score for anomalous packets': results_df.loc[results_df['y_test'] == 1, 'results'].mean(),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1}
