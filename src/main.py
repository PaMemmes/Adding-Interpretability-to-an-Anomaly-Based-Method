import argparse
from train import train 
from xg import xg_main
from preprocess import preprocess

if __name__ =='__main__':

    parser = argparse.ArgumentParser('python3 main.py')
    parser.add_argument('file', help='Model to choose: [xg, wgan, gan]', type=str)
    parser.add_argument('trials', help='Number of trials to hyperopt: [0, inf]', type=int)
    parser.add_argument('retraining', help='Number of times the hyperoptimized model should be retrained', type=int)
    parser.add_argument('epochs', help='Number of epochs to train: [0, inf]', type=int)
    args = parser.parse_args()

    if args.file == 'xg':
        preprocess(subset=False)
        xg_main()
    else:
        preprocess(subset=True)
        train(args.file, args.trials, args.retraining, args.epochs)
    