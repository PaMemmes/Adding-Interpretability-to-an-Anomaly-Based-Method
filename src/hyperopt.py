import json

import keras_tuner

from utils.wasserstein import HyperWGAN

def hyperopt(train, val, num_trials):

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())

    num_features = train.x.shape[1]
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=HyperWGAN(num_features, config, discriminator_extra_steps=5, gp_weight=10.0),
        max_trials=num_trials,
        overwrite=True,
        directory="./hyperopt",
        project_name="HyperWGAN",
    )

    tuner.search(
        train,
        validation_data=(val.x, val.y)
        )

    tuner.results_summary()

    return tuner.get_best_hyperparameters(5)[0]


