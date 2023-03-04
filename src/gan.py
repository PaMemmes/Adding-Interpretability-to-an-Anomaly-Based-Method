import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten, Activation, Dropout
from tensorflow.keras import initializers, layers
import pickle

import keras_tuner

import json


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, num_features):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.num_features = num_features
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="Mean Generator Loss")
        self.dis_loss_tracker = tf.keras.metrics.Mean(name="Mean Discriminator Loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.dis_loss_tracker]

    def compile(self, dis_optim, gen_optim, loss_function):
        super().compile()
        self.dis_optimizer = dis_optim
        self.gen_optimizer = gen_optim
        self.loss_function = loss_function
        
    def train_step(self, data):
        x, y = data
        print('X.SHAPE', x.shape)
        data = tf.convert_to_tensor(x)
        batch_size = x.shape[0]
        noise = tf.random.normal(shape=(batch_size, self.num_features))

        generated_data = self.generator(noise)

        X = tf.concat([generated_data, data], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        
        # Adding noise for labels: https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            preds = self.discriminator(X)
            dis_loss = self.loss_function(labels, preds)

        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        noise = tf.random.normal(shape=[batch_size, self.num_features])
        y_gen = np.zeros((batch_size,1))

        with tf.GradientTape() as tape:
            preds = self.discriminator(self.generator(noise))
            gen_loss = self.loss_function(y_gen, preds)
        grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.gen_loss_tracker.update_state(gen_loss)
        self.dis_loss_tracker.update_state(dis_loss)

        return {
                "g_loss": self.gen_loss_tracker.result(),
                "d_loss": self.dis_loss_tracker.result()
            }

    def test_step(self, data):
        x, y = data
        nr_batches_test = 1
        preds = self.discriminator(x, training=False)
        loss = self.loss_function(y, preds)
        self.dis_loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}


class HyperGAN(keras_tuner.HyperModel):
    def __init__(self, num_features, config):
        super(HyperGAN, self).__init__()
        self.num_features = num_features
        self.config = config

    def get_discriminator(self, dropout):

        discriminator = Sequential()

        discriminator.add(Dense(256, input_dim=self.num_features,
                        kernel_initializer=initializers.glorot_normal(seed=32)))

        for  _, layer in self.config['dis_layers'].items():
            discriminator.add(Dense(layer))
            activation = getattr(tf.keras.layers, self.config['dis_activation'])()
            discriminator.add(activation)

        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))

        return discriminator


    def get_generator(self, activation_function):
        generator = Sequential()
        generator.add(Dense(64, input_dim=self.num_features,
                    kernel_initializer=initializers.glorot_normal(seed=32)))
        generator.add(activation_function)

        for _, layer in self.config['gen_layers'].items():
            generator.add(Dense(layer))
            generator.add(activation_function)

        generator.add(Dense(num_features))
        generator.add(activation_function)

        return generator


    def build(self, hp):
        drop_rate = hp.Float('Dropout', min_value = 0, max_value = 0.30)
        activation_function = hp.Choice('activation function', ['relu', 'leaky_relu', 'tanh'])


        activation_dict = {
            'leaky_relu': layers.LeakyReLU(), 
            'relu': layers.ReLU(),
            'tanh': Activation('tanh')
        }

        self.discriminator = self.get_discriminator(drop_rate)
        self.generator = self.get_generator(activation_dict[activation_function])


        model_gan = GAN(self.discriminator, self.generator, self.num_features)

        optimizer = tf.keras.optimizers.legacy.Adam()
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy()
        model_gan.compile(optimizer, optimizer, binary_crossentropy)
        return model_gan


    def score(self, y, y_pred):
        return 1


    def fit(self, hp, model, data, callbacks=None, **kwargs):
        print('data', data)
        x,y = data.x, data.y
        model.fit(x, y, batch_size=data.batch_size, **kwargs)
        preds = model.discriminator.predict(x)

        score = self.score(y, preds)
        return (model.dis_loss_tracker.result().numpy() + model.gen_loss_tracker.result().numpy()) / 2


if __name__ == '__main__':
    filename = '../data/preprocessed_data.pickle'

    input_file = open(filename, 'rb')
    preprocessed_data = pickle.load(input_file)
    input_file.close()

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())


    dataset = preprocessed_data['dataset']
    print(dataset['train'].x.shape)
    num_features = dataset['train'].x.shape[1]
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

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

    # hypermodel.fit(best_hp, model, train_x, train_y)
    hypermodel.fit(best_hp, model, train)
