import argparse
from train import train 
from xg import xg_main
from preprocess import DataFrame
from interpretable import interpret_tree
import tensorflow as tf
import collections
from math import floor
BATCH_SIZE = 256
FILENAME = 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'
# python3 main.py combined Y 1 1 1

if __name__ =='__main__':
    parser = argparse.ArgumentParser('python3 main.py')
    parser.add_argument('file', help='Model to choose: [xg, wgan, gan, combined]', type=str)
    parser.add_argument('frag_data', help='Use fragmented data to train? Y/N', type=str)
    parser.add_argument('trials', help='Number of trials to hyperopt: [0, inf]', type=int)
    parser.add_argument('retraining', help='Number of times the hyperoptimized model should be retrained [0, inf]', type=int)
    parser.add_argument('epochs', help='Number of epochs to train: [0, inf]', type=int)
    args = parser.parse_args()

    data = DataFrame()
    if args.file == 'xg':
        if args.frag_data == 'Y':
            data.preprocess_anomalies_only_frags(filename=FILENAME, kind=None, frags=True)
            model = xg_main(train=data.train_sqc, test=data.test_sqc, frags=data.test_frag_sqc, trials=args.trials, save='train_w_frags_xg')
            interpret_tree(model, data, save='train_w_frags_xg')
        else:
            data.preprocess(filename=FILENAME)
            model = xg_main(train=data.train_sqc, test=data.test_sqc, frags=data.test_frag_sqc, trials=args.trials, save='train_wo_frags_xg')
            interpret_tree(model, data, save='train_wo_frags_xg')
    elif args.file == 'combined':
        data.preprocess_anomalies_only_frags(filename=FILENAME,kind=None, frags=False)
        model = train('wgan', data.train_sqc, data.test_sqc, None, args.trials, args.retraining, args.epochs, save='combined_wgan')
        gan_data = []
        for i in range(floor(len(data.train_sqc.x) / BATCH_SIZE / 10)):
            noise = tf.random.normal(shape=(BATCH_SIZE, data.train_sqc.x.shape[1]))
            fake_x = model.generator(noise, training=False)
            gan_data.append(fake_x)

        if args.frag_data =='Y':
            data.preprocess_anomalies_only_frags(filename=FILENAME, kind=None, frags=True, add=gan_data)
            model = xg_main(train=data.train_sqc, test=data.test_sqc, frags=data.test_frag_sqc, trials=args.trials, save='train_w_frags_combined')
            interpret_tree(model, data, save='train_w_frags_combined')
        else:
            data.preprocess_anomalies_only_frags(filename=FILENAME, kind=None, frags=False, add=gan_data)
            model = xg_main(train=data.train_sqc, test=data.test_sqc, frags=data.test_frag_sqc, trials=args.trials, save='train_wo_frags_combined')
            interpret_tree(model, data, save='train_wo_frags_combined')

    elif args.file == 'wgan' or args.file == 'gan':
        if args.frag_data =='Y':
            data.preprocess(filename=FILENAME, kind='normal', frags=True, add=None)
            train(args.file, data, args.trials, args.retraining, args.epochs, save='train_w_frags_' + args.file)
        else:
            data.preprocess(filename=FILENAME, kind='normal',frags=False, add=None)
            train(args.file, data, args.trials, args.retraining, args.epochs, save='train_wo_frags_' + args.file)

