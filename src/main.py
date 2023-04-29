import argparse
from train import train 
from xg import xg_main
from preprocess import preprocess, get_frags
from interpretable import interpret_tree
import tensorflow as tf
import collections
from math import floor
BATCH_SIZE = 256

# python3 main.py combined Y 1 1 1

if __name__ =='__main__':

    parser = argparse.ArgumentParser('python3 main.py')
    parser.add_argument('file', help='Model to choose: [xg, wgan, gan, combined]', type=str)
    parser.add_argument('frag_data', help='Use fragmented data? Y/N', type=str)
    parser.add_argument('trials', help='Number of trials to hyperopt: [0, inf]', type=int)
    parser.add_argument('retraining', help='Number of times the hyperoptimized model should be retrained [0, inf]', type=int)
    parser.add_argument('epochs', help='Number of epochs to train: [0, inf]', type=int)
    args = parser.parse_args()

    if args.file == 'xg':
        if args.frag_data == 'Y':
            data = preprocess(kind=None, additional=None, frag_data=True)
            _, frags, _= get_frags()
            xg_main(train=data.train_sqc, test=data.test_sqc, frags=frags, trials=args.trials, save='train_w_frags_xg')
        else:
            data = preprocess(kind=None, additional=None, frag_data=False)
            _, frags, _= get_frags()
            xg_main(train=data.train_sqc, test=data.test_sqc, frags=frags, trials=args.trials, save='train_wo_frags_xg')
        
    elif args.file =='combined':
        data = preprocess(kind='anomaly')
        model = train('wgan', data.train_sqc, data.test_sqc, None, args.trials, args.retraining, args.epochs, save='combined_wgan')
        gan_data = []
        for i in range(floor(len(data.train_sqc.x) / BATCH_SIZE / 10)):
            noise = tf.random.normal(shape=(BATCH_SIZE, data.train_sqc.x.shape[1]))
            fake_x = model.generator(noise, training=False)
            gan_data.append(fake_x)

        if args.frag_data =='Y':
            data = preprocess(kind=None, additional=gan_data, frag_data=True)
            _, frags, _= get_frags()
            model = xg_main(train=data.train_sqc, test=data.test_sqc, frags=frags, trials=args.trials, save='train_w_frags_combined')
        else:
            data = preprocess(kind=None, additional=gan_data, frag_data=False)
            _, frags, _= get_frags()
            model = xg_main(train=data.train_sqc, test=data.test_sqc, frags=frags, trials=args.trials, save='train_wo_frags_combined')

        interpret_tree(model, data.train_sqc, data.test_sqc, data.df_cols, save=args.file)
    elif args.file == 'wgan' or args.file == 'gan':
        if args.frag_data =='Y':
            data = preprocess(kind='normal', additional=None, frag_data=True)
            _, frags, _= get_frags()
            train(args.file, data.train_sqc, data.test_sqc, frags, args.trials, args.retraining, args.epochs, save='train_w_frags_' + args.file)

        else:
            data = preprocess(kind='normal', additional=None, frag_data=False)
            _, frags, _= get_frags()
            train(args.file, data.train_sqc, data.test_sqc, frags, args.trials, args.retraining, args.epochs, save='train_wo_frags_' + args.file)

