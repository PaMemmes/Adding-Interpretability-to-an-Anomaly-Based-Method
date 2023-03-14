import json
import pickle

import keras_tuner

from utils.wasserstein import HyperWGAN

def hyperopt(num_trials):
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
        max_trials=num_trials,
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


