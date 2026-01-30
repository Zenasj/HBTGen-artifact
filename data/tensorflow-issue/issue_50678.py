import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from __future__ import print_function, division

import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras.layers
from tensorflow.keras.layers import BatchNormalization, Bidirectional, LSTM
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers.merge import _Merge


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated trajectory samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((1, 144, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def gradient_penalty_loss(averaged_samples):
    def loss(y_true, y_pred):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    return loss


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class WGANGP():
    def __init__(self):
        self.max_length = 144
        self.features = 1
        self.traj_shape = (self.max_length, self.features)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_discriminator = 5
        optimizer = RMSprop(learning_rate=0.00005)

        # Build the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # -------------------------------
        # Construct Computational Graph
        #       for the Discriminator
        # -------------------------------

        # Freeze generator's layers while training discriminator
        self.generator.trainable = False

        # Trajectory input (real sample)
        real_traj = Input(shape=self.traj_shape)

        # Noise input
        noise_d = Input(shape=(self.latent_dim,))
        # Generate trajectory based of noise (fake sample)
        fake_traj = self.generator(noise_d)

        # Discriminator determines validity of the real and fake trajectories
        fake = self.discriminator(fake_traj)
        valid = self.discriminator(real_traj)

        # Construct weighted average between real and fake trajectories
        interpolated_traj = RandomWeightedAverage()([real_traj, fake_traj])
        # Determine validity of weighted sample
        validity_interpolated = self.discriminator(interpolated_traj)

        self.discriminator_model = Model(inputs=[real_traj, noise_d],
                                         outputs=[valid, fake, validity_interpolated])
        self.discriminator_model.compile(loss=[wasserstein_loss,
                                               wasserstein_loss,
                                               gradient_penalty_loss(averaged_samples=interpolated_traj)],
                                         optimizer=optimizer,
                                         loss_weights=[1, 1, 10])
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the discriminator's layers
        self.discriminator.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        noise_gen = Input(shape=(self.latent_dim,))
        # Generate trajectory based of noise
        traj = self.generator(noise_gen)
        # Discriminator determines validity
        valid = self.discriminator(traj)
        # Defines generator model
        self.generator_model = Model(noise_gen, valid)
        self.generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.traj_shape), activation='tanh'))
        model.add(Reshape(self.traj_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(LSTM(512, input_shape=self.traj_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='tanh'))

        model.summary()

        traj = Input(shape=self.traj_shape)
        validity = model(traj)

        return Model(traj, validity)

    def train(self, epochs, batch_size, sample_interval=50):
        # Training data
        X_train = np.load('data/preprocessed/train.npy', allow_pickle=True)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_discriminator):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of trajectories
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                trajs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the discriminator
                #TODO: Error appears here after the first iteration
                d_loss = self.discriminator_model.train_on_batch([trajs, noise],
                                                                 [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=2000, batch_size=64, sample_interval=10)

vae.add_loss(vae_custom_loss(inputs,outputs))
vae.compile(optimizer=optimiser)