from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

class Generator:
    def __init__(self, latent_dim=5, seq_length=30, batch_size=28, hidden_size=100, num_generated_features=1):
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_generated_features = num_generated_features

        # self.model = tf.keras.models.Sequential([
        #     LSTM(self.hidden_size, input_shape=(self.seq_length, self.latent_dim), return_sequences=True),
        #     tf.keras.layers.Dense(1, input_shape=[None, self.hidden_size]),
        #     tf.keras.layers.Activation('tanh'),
        #     Reshape(target_shape=(self.batch_size, self.seq_length, self.num_generated_features))
        # ])
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(self.hidden_size, input_shape=(
                self.seq_length, self.latent_dim), return_sequences=True, name='g_lstm1'),
            tf.keras.layers.LSTM(
                self.hidden_size, return_sequences=True, recurrent_dropout=0.4, name='g_lstm2'),
            tf.keras.layers.LSTM(1, return_sequences=True, name='g_lstm3')
        ], name='generator')


class Discriminator:
    def __init__(self, input_shape, hidden_size=100):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                hidden_size, input_shape=input_shape, return_sequences=True, name='d_lstm'),
            tf.keras.layers.LSTM(
                hidden_size, return_sequences=True, name='d_lstm2', recurrent_dropout=0.4),
            tf.keras.layers.Dense(1, activation='linear', name='d_output')
        ], name='discriminator')

        self.model.compile(
            loss=self.d_loss, optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['acc'])

    def d_loss(self, y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=True)
        return loss


class GAN:
    real_loss = []
    fake_loss = []
    def __init__(self, *args, **kwargs):

        self.generator = Generator(*args, **kwargs)
        gen_output = (self.generator.seq_length,
                      self.generator.num_generated_features)
        self.discriminator = Discriminator(input_shape=gen_output)
        self.discriminator.model.trainable = False

        self.batch_size = self.generator.batch_size
        self.seq_length = self.generator.seq_length

        self.model = tf.keras.models.Sequential([
            self.generator.model,
            self.discriminator.model
        ], name='gan')

        self.model.compile(
            loss=self.gan_loss, optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['acc'])

    def train(self, epochs, n_eval, d_train_steps=5, load_weights=False, metric='loss'):
        for epoch in range(epochs):
            start = time.time()

            for step in range(steps_over_data):
                tmp_r, tmp_f = [], []

                for _ in range(d_train_steps):

                    x_r, y_r = self.generator.real_samples()
                    x_f, y_f = self.generator.fake_samples()

                    real = self.discriminator.model.fit(
                        x_r, y_r, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=True).history
                    fake = self.discriminator.model.fit(
                        x_f, y_f, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=True).history

                    tmp_r.append(real[metric])
                    tmp_f.append(fake[metric])

            self.real_loss.append(np.mean(tmp_r))
            self.fake_loss.append(np.mean(tmp_f))

            x_gan = self.generator.sample_latent_space()
            y_gan = np.ones((self.batch_size, self.seq_length,
                             self.generator.num_generated_features)).astype(np.float32)

            self.model.fit(
                x_gan, y_gan, batch_size=self.batch_size, epochs=1, verbose=0)

if __name__ == '__main__':
    gan = GAN(latent_dim=5, seq_length=30, batch_size=128)
    gan.discriminator.model.summary()
    gan.load_weights()

    # crashes around epoch ~35
    gan.train(epochs=40, n_eval=1, d_train_steps=3,
              load_weights=True, metric='loss')