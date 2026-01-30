import random
from tensorflow import keras
from tensorflow.keras import optimizers

python
def dice_coef(y_true, y_pred, smooth=1e-7):
    _, W, H, D, C = y_pred.get_shape()

    y_pred = tf.reshape(y_pred, shape=[-1, W * H * D * C])
    y_true = tf.reshape(y_true, shape=[-1, W * H * D * C])

    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)

    dice_numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=1) + smooth
    dice_denominator = tf.reduce_sum(y_pred + y_true, axis=1) + smooth

    dice_coefficient = dice_numerator / dice_denominator
    dice_coefficient = tf.reduce_mean(dice_coefficient)
    return dice_coefficient

import numpy as np
import os
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf
from tensorflow.keras import layers

load_model = 1

latent_dim = 8
learning_rate = 1e-4

BATCH_SIZE = 100
TEST_BATCH_SIZE = 10

color_channels = 1
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images[:5000,::]
test_images = test_images[:1000,::]
n_trainsamples = np.shape(train_images)[0]
n_testsamples = np.shape(test_images)[0]
imsize = np.shape(train_images)[1]
np.random.shuffle(train_images)
train_images = train_images.reshape(-1, imsize, imsize, 1).astype('float32')
test_images = test_images.reshape(-1, imsize, imsize, 1).astype('float32')
train_images /= 255.
test_images /= 255.
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(n_trainsamples).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images)).shuffle(n_testsamples).batch(TEST_BATCH_SIZE)

strategy = tf.distribute.MirroredStrategy()


with strategy.scope():

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    class Sampling(tf.keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding an image."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    original_dim = 784
    intermediate_dim = 64
    latent_dim = 32

    # Define encoder model.
    original_inputs = tf.keras.Input(shape=(imsize, imsize, color_channels), name='encoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(original_inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()((z_mean, z_log_var))
    encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name='encoder')

    # Define decoder model.
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    outputs = tf.keras.layers.Reshape(target_shape=(imsize, imsize, color_channels))(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

    # Define VAE model.
    outputs = decoder(z)
    vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name='vae')

    # Add KL divergence regularization loss.
    kl_loss = - 0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae.add_loss(kl_loss)

    # Train.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    vae.fit(train_dataset, epochs=n_trainsamples//BATCH_SIZE)