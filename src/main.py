import argparse
from train import train 
from xg import xg_main
from preprocess import preprocess
from interpretable import interpret_tree
import tensorflow as tf
import collections
BATCH_SIZE = 256
NUM_GEN_DATA = 500


if __name__ =='__main__':

    parser = argparse.ArgumentParser('python3 main.py')
    parser.add_argument('file', help='Model to choose: [xg, wgan, gan, combined]', type=str)
    parser.add_argument('trials', help='Number of trials to hyperopt: [0, inf]', type=int)
    parser.add_argument('retraining', help='Number of times the hyperoptimized model should be retrained', type=int)
    parser.add_argument('epochs', help='Number of epochs to train: [0, inf]', type=int)
    args = parser.parse_args()

    if args.file == 'xg':
        preprocess(kind=None)
        xg_main('xg')
    elif args.file =='combined':
        train_data = preprocess(kind='anomaly')
        model = train('wgan', args.trials, args.retraining, args.epochs, save='combined')
        gen_data = []
        for i in range(NUM_GEN_DATA):
            noise = tf.random.normal(shape=(BATCH_SIZE, train_data.x.shape[1]))
            fake_x = model.generator(noise, training=False)
            gen_data.append(fake_x)
        preprocess(kind=None, add_data=gen_data)
        normals = collections.Counter(train_data.y)[0]
        anomalies = collections.Counter(train_data.y)[1]
        anom_percent = anomalies / (normals + anomalies)
        model = xg_main(save='combined', feature_weights=[1-anom_percent, anom_percent])
        interpret_tree(model, save='combined')
    elif args.file == 'wgan' or args.file == 'gan':
        preprocess(kind='normal')
        train(args.file, args.trials, args.retraining, args.epochs, save=args.file)
