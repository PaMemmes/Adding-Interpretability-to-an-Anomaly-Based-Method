from pathlib import Path
import json
import collections
import pickle

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score

from hyperopt import hyperopt
from utils.utils import test_model
from utils.wasserstein import HyperWGAN
from utils.plots import plot_confusion_matrix, plot_roc, plot_precision_recall

NUM_TRIALS = 3
NUM_RETRAINING = 5
FILENAME = '../data/preprocessed_data.pickle'

if __name__ == '__main__':

    input_file = open(FILENAME, 'rb')
    preprocessed_data = pickle.load(input_file)
    input_file.close()

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())

    dataset = preprocessed_data['test_dataset']
    num_features = dataset['train'].x.shape[1]
    train = dataset['train']
    test = dataset['test']

    best_hp = hyperopt(NUM_TRIALS)
    
    for i in range(NUM_RETRAINING):
        name = '../experiments/experiment' + str(i) + '_tuner'
        Path(name).mkdir(parents=True, exist_ok=True)
        hypermodel = HyperWGAN(num_features, config, discriminator_extra_steps=3, gp_weight=10.0)
        model = hypermodel.build(best_hp)

        hypermodel.fit(best_hp, model, train)

        results_df, results = test_model(hypermodel, test)

        normals = collections.Counter(test.y)[0]
        anomalies = collections.Counter(test.y)[1]
        anomalies_percentage = anomalies / (normals + anomalies)

        # Obtaining the lowest "anomalies_percentage" score
        per = np.percentile(results, anomalies_percentage*100)
        y_pred = results.copy()
        y_pred = np.array(y_pred)
        probas = np.vstack((1-y_pred, y_pred)).T

        # Thresholding based on the score
        inds = y_pred > per
        inds_comp = y_pred <= per
        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(
            test.y, y_pred, average='binary')
        accuracy = accuracy_score(test.y, y_pred)

        fpr, tpr, thresholds = roc_curve(test.y, y_pred)
        auc_curve = auc(fpr, tpr)
        cm = confusion_matrix(test.y, y_pred)
        
        plot_confusion_matrix(cm, name + '/confusion_gan_only_cic.png', 'GAN')
        plot_roc(tpr, fpr, auc_curve, name + '/roc_gan_only_cic.png', 'GAN')
        plot_precision_recall(test.y, probas, name + '/precision_recall_only_cic.png')
        results = {
                'Normals (%)': 1 - anomalies_percentage,
                'Anomalies (%)': anomalies_percentage,
                'Mean Score for normal packets': results_df.loc[results_df['y_test'] == 0, 'results'].mean(),
                'Mean Score for anomalous packets': results_df.loc[results_df['y_test'] == 1, 'results'].mean(),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Best HP: ': best_hp['Dropout']
        }
        print(results)
