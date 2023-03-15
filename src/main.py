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

NUM_TRIALS = 2
NUM_RETRAINING = 5
FILENAME = '../data/preprocessed_data.pickle'

if __name__ == '__main__':

    input_file = open(FILENAME, 'rb')
    preprocessed_data = pickle.load(input_file)
    input_file.close()

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())

    train_dataset = preprocessed_data['dataset']
    train = train_dataset['train']
    val = train_dataset['val']
    best_hp = hyperopt(train, val, NUM_TRIALS)


    test_dataset = preprocessed_data['test_dataset']
    num_features = test_dataset['train'].x.shape[1]
    train = test_dataset['train']
    test = test_dataset['test']

    saves = []
    
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
        per = np.percentile(results, 0.1*100)
        y_pred = results.copy()
        y_pred = np.array(y_pred)
        y_pred2 = y_pred.copy()
        probas = np.vstack((1-y_pred, y_pred)).T

        # Thresholding based on the score
        inds = y_pred > per
        inds_comp = y_pred <= per
        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(test.y, y_pred, average='binary')
        accuracy = accuracy_score(test.y, y_pred)

        fpr, tpr, thresholds = roc_curve(test.y, y_pred)
        
        
        # gmean = np.sqrt(tpr * (1 - fpr))
        # # Find the optimal threshold
        # index = np.argmax(gmean)
        # thresholdOpt = round(thresholds[index], ndigits = 4)
        # gmeanOpt = round(gmean[index], ndigits = 4)
        # fprOpt = round(fpr[index], ndigits = 4)
        # tprOpt = round(tpr[index], ndigits = 4)
        # print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
        # print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

        # inds = y_pred2 > thresholdOpt
        # inds_comp = y_pred2 <= thresholdOpt
        # y_pred2[inds] = 0
        # y_pred2[inds_comp] = 1

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
        saves.append(results)
    print('Best result: ', sorted(saves, key=lambda d: d['Accuracy'])[-1])
