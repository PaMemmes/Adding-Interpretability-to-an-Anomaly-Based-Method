from pathlib import Path
import json
import collections
import pickle

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score

from hyperopt import hyperopt
from utils.utils import test_model
from utils.gan import HyperGAN
from utils.plots import plot_confusion_matrix, plot_roc, plot_precision_recall

NUM_TRIALS = 2
NUM_RETRAINING = 10
FILENAME = '../data/preprocessed_data.pickle'

if __name__ == '__main__':

    input_file = open(FILENAME, 'rb')
    preprocessed_data = pickle.load(input_file)
    input_file.close()

    with open('configs/config.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())

    dataset = preprocessed_data['dataset']
    train = dataset['train']
    test = dataset['test']
    best_hp = hyperopt(train, test, NUM_TRIALS)


    num_features = dataset['train'].x.shape[1]
    saves = []
    
    for i in range(NUM_RETRAINING):
        name = '../experiments/gan_experiment' + str(i) + '_tuner'
        Path(name).mkdir(parents=True, exist_ok=True)
        hypermodel = HyperGAN(num_features, config)
        model = hypermodel.build(best_hp)

        hypermodel.fit(best_hp, model, train, epochs=50)

        results_df, results = test_model(hypermodel, test)

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

        precision, recall, f1, _ = precision_recall_fscore_support(test.y, y_pred, average='binary')
        accuracy = accuracy_score(test.y, y_pred)
        fpr, tpr, thresholds = roc_curve(test.y, probas[:,0])
        
        auc_val = auc(fpr, tpr)
        cm = confusion_matrix(test.y, y_pred)
        
        plot_confusion_matrix(cm, name + '/confusion_gan_only_cic.png', 'GAN')
        plot_roc(tpr, fpr, auc_val, name + '/roc_gan_only_cic.png', 'GAN')
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
                'AUC': auc_val,
                'Best HP: ': best_hp['Dropout'],
                'I': i
        }
        print(results)
        saves.append(results)
    print('Best result: ', sorted(saves, key=lambda d: d['Accuracy'])[-1])
