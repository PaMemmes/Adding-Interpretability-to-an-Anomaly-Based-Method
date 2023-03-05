import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten, Activation, Dropout
from tensorflow.keras import initializers, layers
import pickle

import keras_tuner
import json

from utils.network import HyperGAN

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

    tuner = keras_tuner.BayesianOptimization(
        hypermodel=HyperGAN(num_features, config),
        max_trials=2,
        overwrite=True,
        directory="./experiments",
        project_name="HyperGAN",
    )

    tuner.search(
        train,
        validation_data=(val.x, val.y)
        )

    tuner.results_summary()

    best_hp = tuner.get_best_hyperparameters(5)[0]

    hypermodel = HyperGAN(num_features, config)
    model = hypermodel.build(best_hp)

    hypermodel.fit(best_hp, model, train)
