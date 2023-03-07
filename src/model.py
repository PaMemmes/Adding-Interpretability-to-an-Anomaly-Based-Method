import pickle
import collections
import json
from pathlib import Path

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

from tqdm import tqdm

from utils.plots import plot_confusion_matrix, plot_roc, plot_losses, plot_precision_recall
from utils.network import get_discriminator, get_generator, make_gan_network

filename = '../data/preprocessed_data.pickle'

if __name__ =='__main__':
    for i in range(1,10):
        name = '../experiments/experiment' + str(i)
        json_file = name + '/experiment' + str(i) + '.json'
        Path(name).mkdir(parents=True, exist_ok=True)

        input_file = open(filename, 'rb')
        preprocessed_data = pickle.load(input_file)
        input_file.close()
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.loads(f.read())

        dataset = preprocessed_data['dataset']

        num_features = dataset['x_train'].shape[1]

        normals = collections.Counter(dataset['y_test'])[0]
        anomalies = collections.Counter(dataset['y_test'])[1]
        anomalies_percentage = anomalies / (normals + anomalies)
        print('Number of Normal Network packets in the test set:', normals)
        print('Number of Anomalous Network packets in the test set:', anomalies)
        print('Ratio of anomalous to normal network packets in the test set: ', anomalies_percentage)

        x_train, y_train, x_test, y_test = dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']

        batch_count = x_train.shape[0] // config['batch_size']
        pbar = tqdm(total=config['epochs'] * batch_count, position=0, leave=True)

        gan_loss = []
        discriminator_loss = []
        generator_loss = []

        generator = get_generator(config, num_features)
        discriminator = get_discriminator(config, num_features)
        gan = make_gan_network(config, discriminator, generator, input_dim=num_features)

        print("Number params: ", gan.count_params())

        for epoch in range(config['epochs']):
            for index in range(batch_count):
                pbar.update(1)
                noise = np.random.normal(0, 1, size=[config['batch_size'], num_features])

                generated_images = generator.predict_on_batch(noise)

                image_batch = x_train[index * config['batch_size']: (index + 1) * config['batch_size']]

                X = np.vstack((generated_images, image_batch))
                y_dis = np.ones(2*config['batch_size'])
                y_dis[:config['batch_size']] = 0

                discriminator.trainable = True
                d_loss = discriminator.train_on_batch(X, y_dis)

                noise = np.random.uniform(0, 1, size=[config['batch_size'], num_features])
                y_gen = np.ones(config['batch_size'])
                discriminator.trainable = False
                g_loss = gan.train_on_batch(noise, y_gen)

                discriminator_loss.append(d_loss)
                generator_loss.append(g_loss)
                gan_loss.append(g_loss)

            print(f"Epoch {epoch} Batch {index} {batch_count} [D loss: {d_loss}] [G loss:{g_loss}]")
        

        plot_losses(discriminator_loss, generator_loss, name + '/loss_gan.png')

        nr_batches_test = np.ceil(x_test.shape[0] // config['batch_size']).astype(np.int32)

        results = []

        for t in range(nr_batches_test + 1):
            ran_from = t * config['batch_size']
            ran_to = (t + 1) * config['batch_size']
            image_batch = x_test[ran_from:ran_to]
            tmp_rslt = discriminator.predict(x=image_batch, batch_size=128, verbose=0)
            results = np.append(results, tmp_rslt)
        pd.options.display.float_format = '{:20,.7f}'.format
        results_df = pd.concat([pd.DataFrame(results), pd.DataFrame(y_test)], axis=1)
        results_df.columns = ['results', 'y_test']
        print('Mean score for normal packets :',
            results_df.loc[results_df['y_test'] == 0, 'results'].mean())
        print('Mean score for anomalous packets :',
            results_df.loc[results_df['y_test'] == 1, 'results'].mean())

        # Obtaining the lowest "anomalies_percentage" score
        per = np.percentile(results, anomalies_percentage*100)
        y_pred = results.copy()
        y_pred = np.array(y_pred)

        probas = np.vstack((1-y_pred, y_pred)).T
        plot_precision_recall(y_test, probas, name + '/precision_recall_only_cic.png')
        
        # Thresholding based on the score
        inds = y_pred > per
        inds_comp = y_pred <= per
        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary')
        print('Accuracy Score :', accuracy_score(y_test, y_pred))
        print('Precision :', precision)
        print('Recall :', recall)
        print('F1 :', f1)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_curve = auc(fpr, tpr)
        plot_roc(tpr, fpr, auc_curve, name + '/roc_gan_only_cic.png', 'GAN')
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, name + '/confusion_gan_only_cic.png', 'GAN')
        
        results = {
                'Normals (%)': 1 - anomalies_percentage,
                'Anomalies (%)': anomalies_percentage,
                'Mean Score for normal packets': results_df.loc[results_df['y_test'] == 0, 'results'].mean(),
                'Mean Score for anomalous packets': results_df.loc[results_df['y_test'] == 1, 'results'].mean(),
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision,
                'Recall': recall,
                'F1': f1}
        save_results(json_file, config, results)
