from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import os
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers.distribution_layer
import tensorflow as tf


load_model = 1

latent_dim = 8
learning_rate = 1e-4

BATCH_SIZE = 100
TEST_BATCH_SIZE = 10


color_channels = 1
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images[:5000,::]
n_trainsamples = np.shape(train_images)[0]
imsize = np.shape(train_images)[1]
train_images = train_images.reshape(-1, imsize, imsize, 1).astype('float32')
train_images /= 255.
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_images)).shuffle(n_trainsamples).repeat().batch(BATCH_SIZE)

image_input = tf.keras.Input(shape=(imsize, imsize, color_channels), name='encoder_input')
x = tf.keras.layers.Flatten()(image_input)
x = tf.keras.layers.Dense(500, activation='softplus', name="Inference-l1_Dense")(x)
x = tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim))(x)
z = tfpl.MultivariateNormalTriL(latent_dim)(x)
prior = tfd.Independent(tfd.Normal(loc=[0., 0], scale=1), reinterpreted_batch_ndims=1)
tfpl.KLDivergenceAddLoss(prior, weight=1.0)

encoder = tf.keras.Model(inputs=image_input, outputs=z, name='encoder')

latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
x = tf.keras.layers.Dense(500, activation='softplus', name="Generative-l1_Dense")(latent_inputs)
x = tf.keras.layers.Dense(imsize ** 2 * color_channels, activation='sigmoid', name="Generative-l3_Dense_out")(x)  

output_probs = tf.keras.layers.Reshape(target_shape=(imsize, imsize, color_channels), name="Generative-output_probs")(x)

decoder = tf.keras.Model(inputs=latent_inputs, outputs=output_probs, name='decoder')
output_probs = decoder(z)

vae_model = tf.keras.Model(inputs=image_input, outputs=output_probs, name='vae')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae_model.compile(optimizer, tf.keras.losses.BinaryCrossentropy())

vae_model.fit(train_dataset, steps_per_epoch=n_trainsamples // BATCH_SIZE, epochs=4)

latents = encoder.predict(train_images[:4,::])
print('latent shape: ' + latents.shape())