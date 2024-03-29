import argparse
from train import train
from xg import xg_main
from preprocess import DataFrame
from interpretable import interpret_tree
import tensorflow as tf
import collections
from math import floor

BATCH_SIZE = 256


def run_xg(trials, frags=True):
    # Runs xg boost model and interprets it
    if frags:
        name = 'xg_frags'
    else:
        name = 'xg'
    data = DataFrame()
    data.preprocess(filename=None, kind=None, frags=name, scale=False)
    model = xg_main(
        train=data.train_sqc,
        test=data.test_sqc,
        frags=data.test_frag_sqc,
        trials=trials,
        save=name)
    interpret_tree(model, data, save=name)


def run_nn(model_name, epochs, retrain, trials, frags=True):
    # Runs neural network: model_name = [gan, wgan]
    if frags:
        name = '_frags'
    else:
        name = ''
    data = DataFrame()
    data.preprocess(filename=None, kind='normal', frags=frags, add=None)
    train(
        model_name=model_name,
        data=data,
        frags=data.test_frag_sqc,
        trials=trials,
        num_retraining=retrain,
        epochs=args.epochs,
        save=model_name + '_' + name)


def run_combined(epochs, retrain, trials, frags=True):
    # Runs combined model: 1. WGAN 2. XGBoost
    if frags:
        name = '_frags'
    else:
        name = ''
    data = DataFrame()
    data.preprocess(filename=None, kind='anomaly', frags=False)
    model = train(
        'wgan',
        data=data,
        frags=None,
        trials=args.trials,
        num_retraining=args.retraining,
        epochs=args.epochs,
        save='combined_wgan')
    gan_data = []
    for i in range(floor(len(data.train_sqc.x) / BATCH_SIZE / 10)):
        noise = tf.random.normal(shape=(BATCH_SIZE, data.train_sqc.x.shape[1]))
        fake_x = model.generator(noise, training=False)
        gan_data.append(fake_x)
    print('Additional data: ', len(gan_data) * BATCH_SIZE)
    data = DataFrame()
    data.preprocess(
        filename=None,
        kind=None,
        frags=frags,
        add=gan_data,
        scale=False)
    model = xg_main(
        train=data.train_sqc,
        test=data.test_sqc,
        frags=data.test_frag_sqc,
        trials=args.trials,
        save='combined' + name)
    interpret_tree(model, data, save='combined' + name)


if __name__ == '__main__':
    # python main.py 1 1 1
    parser = argparse.ArgumentParser('python3 main.py')
    parser.add_argument(
        'trials',
        help='Number of trials to hyperopt: [0, inf]',
        type=int)
    parser.add_argument(
        'retraining',
        help='Number of times the hyperoptimized model should be retrained [0, inf]',
        type=int)
    parser.add_argument(
        'epochs',
        help='Number of epochs to train: [0, inf]',
        type=int)
    args = parser.parse_args()

    print('Running XG with no frags')
    run_xg(args.trials, frags=False)
    print('Running XG with frags')
    run_xg(args.trials, frags=True)

    print('Running Combined with no frags')
    run_combined(args.epochs, args.retraining, args.trials, frags=False)
    print('Running Combined with frags')
    run_combined(args.epochs, args.retraining, args.trials, frags=True)

    print('Running Gan with no frags')
    run_nn('gan', args.epochs, args.retraining, args.trials, frags=False)
    print('Running Gan with frags')
    run_nn('gan', args.epochs, args.retraining, args.trials, frags=True)

    print('Running Wgan with no frags')
    run_nn('wgan', args.epochs, args.retraining, args.trials, frags=False)
    print('Running Wgan with frags')
    run_nn('wgan', args.epochs, args.retraining, args.trials, frags=True)
