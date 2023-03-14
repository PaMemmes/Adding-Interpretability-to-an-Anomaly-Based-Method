import json
import collections
import pickle

import keras_tuner
import tensorflow as tf
import numpy as np
import pandas as pd

from utils.utils import test_model
from utils.wasserstein import HyperWGAN
from utils.plots import plot_confusion_matrix, plot_roc, plot_losses, plot_precision_recall
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score

def hyperopt():
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

    tuner = keras_tuner.BayesianOptimization(
        hypermodel=HyperWGAN(num_features, config),
        max_trials=25,
        overwrite=True,
        directory="./experiments",
        project_name="HyperWGAN",
    )

    tuner.search(
        train,
        validation_data=(val.x, val.y)
        )

    tuner.results_summary()

    return tuner.get_best_hyperparameters(5)[0]


