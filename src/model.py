import pandas as pd
import numpy as np
import glob
import pickle
import os
import collections

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

#import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten, Activation
from tensorflow.keras.layers import Dropout, multiply, GaussianNoise, MaxPooling2D, concatenate
from tensorflow.keras import initializers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow.keras.optimizers

from tqdm import tqdm
import json

import plots

figure_path = 'model_plots/'
filename = 'data/preprocessed_data.pickle'

# Labels normal data as 0, anomalies as 1
def make_labels_binary(label_encoder, labels):
    normal_data_index = np.where(label_encoder.classes_ == 'BENIGN')[0][0]
    new_labels = labels.copy()
    new_labels[labels != normal_data_index] = 1
    new_labels[labels == normal_data_index] = 0
    return new_labels

def subset_normal(x_train, y_train):
    temp_df = x_train.copy()
    temp_df['label'] = y_train
    temp_df = temp_df.loc[temp_df['label'] == 0]
    y_train = temp_df['label'].copy()
    temp_df = temp_df.drop('label', axis = 1)
    x_train = temp_df.copy()
    return x_train,y_train

def get_generator(config, num_features):
    generator = Sequential()
    generator.add(Dense(64, input_dim=num_features,
                  kernel_initializer=initializers.glorot_normal(seed=32)))
    generator.add(Activation('relu'))

    for _, layer in config['gen_layers'].items():
        print(Activation(config['gen_activation']))
        generator.add(Dense(layer))
        generator.add(Activation(config['gen_activation']))
    
    generator.add(Dense(num_features))
    generator.add(Activation('tanh'))

    optim = getattr(tensorflow.optimizers.legacy, config['optimizer'])(learning_rate=config['learning_rate'], beta_1=config['momentum'])
    generator.compile(loss=config['loss'], optimizer=optim)

    return generator


def get_discriminator(config, num_features):

    discriminator = Sequential()

    discriminator.add(Dense(256, input_dim=num_features,
                      kernel_initializer=initializers.glorot_normal(seed=32)))
    
    for  _, layer in config['dis_layers'].items():
        discriminator.add(Dense(layer))
        activation = getattr(tensorflow.keras.layers, config['dis_activation'])()
        discriminator.add(activation)
    
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))
    
    optim = getattr(tensorflow.optimizers.legacy, config['optimizer'])(learning_rate=config['learning_rate'], beta_1=config['momentum'])
    discriminator.compile(loss=config['loss'], optimizer=optim)

    return discriminator

def make_gan_network(discriminator, generator, input_dim):
    discriminator.trainable = False
    gan_input = Input(shape=(input_dim,))
    print('gan_input', gan_input.shape)
    x = generator(gan_input)
    print('x', x.shape)
    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=gan_output)
    optim = getattr(tensorflow.optimizers.legacy, config['optimizer'])(learning_rate=config['learning_rate'], beta_1=config['momentum'])
    gan.compile(loss='binary_crossentropy', optimizer=optim)

    return gan


if __name__ =='__main__':
    
    input_file = open(filename, 'rb')
    preprocessed_data = pickle.load(input_file)
    input_file.close()

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())

    le = preprocessed_data['le']
    x_train = preprocessed_data['x_train']
    y_train = preprocessed_data['y_train']
    x_test = preprocessed_data['x_test']
    y_test = preprocessed_data['y_test']

    num_features = x_train.shape[1]

    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == x_test.shape[1]

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    y_train = make_labels_binary(le, y_train)
    y_test = make_labels_binary(le, y_test)

    #Subsetting only Normal Network packets in training set
    x_train, y_train = subset_normal(x_train, y_train)

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    normals = collections.Counter(y_test)[0]
    anomalies = collections.Counter(y_test)[1]
    anomalies_percentage = anomalies / (normals + anomalies)
    print('Number of Normal Network packets in the test set:', normals)
    print('Number of Anomalous Network packets in the test set:', anomalies)
    print('Ratio of anomalous to normal network packets: ', anomalies_percentage)

    x_train, y_train, x_test, y_test = dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']

    batch_count = x_train.shape[0] // config['batch_size']
    pbar = tqdm(total=config['epochs'] * batch_count, position=0, leave=True)

    gan_loss = []
    discriminator_loss = []
    generator_loss = []

    generator = get_generator(config, num_features)
    discriminator = get_discriminator(config, num_features)
    gan = make_gan_network(discriminator, generator, input_dim=num_features)

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

        print("Epoch %d Batch %d/%d [D loss: %f] [G loss:%f]" %
            (epoch, index, batch_count, d_loss, g_loss))
    

    plots.plot_losses(discriminator_loss, generator_loss, gan_loss, figure_path + 'loss_gan.png')

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

    # Thresholding based on the score
    inds = (y_pred > per)
    inds_comp = (y_pred <= per)
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
    plots.plot_roc(tpr, fpr, auc_curve, figure_path + 'roc_gan_only_cic.png', 'GAN')
    cm = confusion_matrix(y_test, y_pred)
    plots.plot_confusion_matrix(cm, figure_path + 'confusion_gan_only_cic.png', 'GAN')