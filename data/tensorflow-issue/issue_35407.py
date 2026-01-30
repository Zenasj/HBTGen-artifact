import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from keras.models import Model
from keras.layers import Lambda, Dense, LSTM, Activation, Input, Bidirectional, Dropout, Reshape, Conv2DTranspose, TimeDistributed, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import _Merge
import librosa
import tensorflow as tf
from functools import partial
import sys
import os
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.latent_dim = 100
        self.d = 64
        self.c = 16
        self.a = 1
        self.Fs = 44100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_audio = Input(shape=(self.a*256*self.d, 1))

        # Noise input
        z_disc = Input(shape=(self.a, self.latent_dim))
        # Generate image based of noise (fake sample)
        fake_audio = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_audio)
        valid = self.critic(real_audio)

        # Construct weighted average between real and fake images
        interpolated_audio = RandomWeightedAverage()([real_audio, fake_audio])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_audio)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_audio)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_audio, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.a, self.latent_dim))
        # Generate images based of noise
        audio = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(audio)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
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


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def apply_phaseshuffle(self, x, rad=2, pad_type='reflect'):
        b, x_len, nch = x.get_shape().as_list()

        phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
        pad_l = tf.maximum(phase, 0)
        pad_r = tf.maximum(-phase, 0)
        phase_start = pad_r
        x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

        x = x[:, phase_start:phase_start+x_len]
        x.set_shape([b, x_len, nch])

        return x

    def build_generator(self):
        d=self.d
        c=self.c
        a=self.a

        # Prelim layers
        input_layer = Input(shape=(a, 100))

        dense_layer0 = TimeDistributed(Dense(256*d, input_shape=(100,)))(input_layer)#
        reshape_layer0 = TimeDistributed(Reshape((c, c*d)))(dense_layer0)#
        relu_layer0 = TimeDistributed(Activation('relu'))(reshape_layer0)#

        # WaveCNN layers
        c //= 2
        expanded_layer0 = TimeDistributed(Lambda(lambda x: K.expand_dims(x, axis=1)))(relu_layer0)#relu_layer1
        conv1d_t_layer0 = TimeDistributed(Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same'))(expanded_layer0)
        slice_layer0 = Lambda(lambda x: x[:, :, 0])(conv1d_t_layer0)
        relu_layer2 = TimeDistributed(Activation('relu'))(slice_layer0)

        c //= 2
        expanded_layer1 = TimeDistributed(Lambda(lambda x: K.expand_dims(x, axis=1)))(relu_layer2)
        conv1d_t_layer1 = TimeDistributed(Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same'))(expanded_layer1)
        slice_layer1 = Lambda(lambda x: x[:, :, 0])(conv1d_t_layer1)
        relu_layer3 = TimeDistributed(Activation('relu'))(slice_layer1)

        c //= 2
        expanded_layer2 = TimeDistributed(Lambda(lambda x: K.expand_dims(x, axis=1)))(relu_layer3)
        conv1d_t_layer2 = TimeDistributed(Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same'))(expanded_layer2)
        slice_layer2 = Lambda(lambda x: x[:, :, 0])(conv1d_t_layer2)
        relu_layer4 = TimeDistributed(Activation('relu'))(slice_layer2)

        c //= 2
        expanded_layer3 = TimeDistributed(Lambda(lambda x: K.expand_dims(x, axis=1)))(relu_layer4)
        conv1d_t_layer3 = TimeDistributed(Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same'))(expanded_layer3)
        slice_layer3 = Lambda(lambda x: x[:, :, 0])(conv1d_t_layer3)
        relu_layer5 = TimeDistributed(Activation('relu'))(slice_layer3)

        expanded_layer4 = TimeDistributed(Lambda(lambda x: K.expand_dims(x, axis=1)))(relu_layer5)
        conv1d_t_layer4 = TimeDistributed(Conv2DTranspose(1, (1, 25), strides=(1, 4), padding='same'))(expanded_layer4)#strides=(1,1)
        slice_layer4 = Lambda(lambda x: x[:, :, 0])(conv1d_t_layer4)
        tanh_layer0 = TimeDistributed(Activation('tanh'))(slice_layer4)

        reshape_layer1 = Reshape((a*256*d, 1))(tanh_layer0)

        model = Model(inputs=input_layer, outputs=reshape_layer1)

        print(model.summary())

        return model

    def build_critic(self):
        d=self.d
        c=self.c
        a=self.a

        input_layer = Input(shape=(a*256*d, 1))#d*d
        reshape_layer0 = Reshape((a, 256*d, 1))(input_layer)

        conv1d_layer0 = TimeDistributed(Conv1D(d, 25, strides=4, padding='same'))(reshape_layer0)#//2
        LReLU_layer0 = TimeDistributed(LeakyReLU(alpha=0.2))(conv1d_layer0)
        phaseshuffle_layer0 = TimeDistributed(Lambda(lambda x: self.apply_phaseshuffle(x)))(LReLU_layer0)

        conv1d_layer1 = TimeDistributed(Conv1D(2*d, 25, strides=4, padding='same'))(phaseshuffle_layer0)#d
        LReLU_layer1 = TimeDistributed(LeakyReLU(alpha=0.2))(conv1d_layer1)
        phaseshuffle_layer1 = TimeDistributed(Lambda(lambda x: self.apply_phaseshuffle(x)))(LReLU_layer1)

        conv1d_layer2 = TimeDistributed(Conv1D(4*d, 25, strides=4, padding='same'))(phaseshuffle_layer1)#2*d
        LReLU_layer2 = TimeDistributed(LeakyReLU(alpha=0.2))(conv1d_layer2)
        phaseshuffle_layer2 = TimeDistributed(Lambda(lambda x: self.apply_phaseshuffle(x)))(LReLU_layer2)

        conv1d_layer3 = TimeDistributed(Conv1D(8*d, 25, strides=4, padding='same'))(phaseshuffle_layer2)#4*d
        LReLU_layer3 = TimeDistributed(LeakyReLU(alpha=0.2))(conv1d_layer3)
        phaseshuffle_layer3 = TimeDistributed(Lambda(lambda x: self.apply_phaseshuffle(x)))(LReLU_layer3)

        conv1d_layer4 = TimeDistributed(Conv1D(16*d, 25, strides=4, padding='same'))(phaseshuffle_layer3)#8*d,strides=4
        LReLU_layer4 = TimeDistributed(LeakyReLU(alpha=0.2))(conv1d_layer4)
        phaseshuffle_layer4 = TimeDistributed(Lambda(lambda x: self.apply_phaseshuffle(x)))(LReLU_layer4)
    
        reshape_layer1 = Reshape((a, 256*d))(phaseshuffle_layer4)
        slice_layer0 = Lambda(lambda x: x[:, 0])(reshape_layer1)#

        dense_layer1 = Dense(1, input_shape=(a, 256*d))(slice_layer0)#dropout_layer1

        model = Model(inputs=input_layer, outputs=dense_layer1)

        print(model.summary())

        return model

    def train(self, epochs, batch_size, sample_interval=50):

        X_train = []

        for file in os.listdir(r"C:\Users\Harry\source\repos\tfworldhackathon\Data"):
            with open(r"C:\Users\Harry\source\repos\tfworldhackathon\Data" + fr"\{file}", "rb") as f:
                samples, _ = librosa.load(f, sr=self.Fs)
                X_train.append(np.array([np.array([sample]) for sample in samples[:self.a*256*self.d]]))
                if "17" in file:
                    break

        X_train = np.array(X_train)

        print(X_train.shape)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                audios = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.a, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([audios, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_audio(epoch)

    def sample_audio(self, epoch):
        print(f"Checkpoint {epoch}")
        noise = np.random.normal(0, 1, (5, self.a, self.latent_dim))
        gen_audios = self.generator.predict(noise)
        for i in range(len(gen_audios)):
            audio = gen_audios[i]
            audio.flatten()
            librosa.output.write_wav(f"output/{epoch}-{i}.wav", audio, sr=self.Fs)


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=8, sample_interval=100)