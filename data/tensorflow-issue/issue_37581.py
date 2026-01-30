import random
from tensorflow import keras
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

  def __init__(self,
               latent_dim=32,
               intermediate_dim=64,
               name='encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_mean = layers.Dense(latent_dim)
    self.dense_log_var = layers.Dense(latent_dim)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.dense_proj(inputs)
    z_mean = self.dense_mean(x)
    z_log_var = self.dense_log_var(x)
    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var, z


class Decoder(layers.Layer):
  """Converts z, the encoded digit vector, back into a readable digit."""

  def __init__(self,
               original_dim,
               intermediate_dim=64,
               name='decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_output = layers.Dense(original_dim, activation='sigmoid')

  def call(self, inputs):
    x = self.dense_proj(inputs)
    return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               original_dim,
               intermediate_dim=64,
               latent_dim=32,
               name='autoencoder',
               **kwargs):
    super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
    self.original_dim = original_dim
    self.encoder = Encoder(latent_dim=latent_dim,
                           intermediate_dim=intermediate_dim)
    self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
    kl_loss = - 0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    self.add_loss(kl_loss)
    return reconstructed


def plotRecons(model, dat, savePlot=None):
  """Plots reconstructions of provided images.
  """
  fig, ax = plt.subplots(len(list(dat)), 2)
  for i, im in enumerate(dat):
    ax[i,0].imshow(im[:,:,0], cmap='gray_r', vmin=0.0, vmax=1.0)
    thisrecon = model(tf.cast(tf.reshape(im, (1,4096)), 'float32'))
    thisrecon = np.reshape(thisrecon, (64,64))
    randomIm = np.random.random(thisrecon.shape)
    #thisrecon = np.array((thisrecon > randomIm), dtype=int)
    ax[i,1].imshow(thisrecon, cmap='gray_r', vmin=0.0, vmax=1.0)
    ax[i,0].tick_params(axis='both', which='both',
                        left=False, right=False, bottom=False, top=False,
                        labelleft=False, labelbottom=False,
                        labelright=False, labeltop=False)
    ax[i,1].tick_params(axis='both', which='both',
                        left=False, right=False, bottom=False, top=False,
                        labelleft=False, labelbottom=False,
                        labelright=False, labeltop=False)

  fig.tight_layout()
  if savePlot is not None:
    fig.savefig(savePlot)
  plt.show()


def trainCustom(train_dataset, epochs=2, original_dim=4096):
  vae = VariationalAutoEncoder(original_dim, 64, 32)
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  #loss_fn = tf.keras.losses.MeanSquaredError()
  loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                               reduction=tf.keras.losses.Reduction.SUM)

  # Iterate over epochs.
  for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
  
    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
      with tf.GradientTape() as tape:
        reconstructed = vae(x_batch_train)
        # Compute reconstruction loss
        loss = loss_fn(x_batch_train, reconstructed)
        loss += sum(vae.losses)  # Add KLD regularization loss
  
      grads = tape.gradient(loss, vae.trainable_weights)
      optimizer.apply_gradients(zip(grads, vae.trainable_weights))
  
      if step % 1000 == 0:
        print('step %s: mean loss = %s' % (step, loss))
  
  return vae


def trainBuiltIn(train_dataset, epochs=2, original_dim=4096):
  vae = VariationalAutoEncoder(original_dim, 64, 32)
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  
  vae.compile(optimizer, #loss=tf.keras.losses.MeanSquaredError())
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                                      reduction=tf.keras.losses.Reduction.SUM))

  zipdata = tf.data.Dataset.zip((train_dataset, train_dataset))

  vae.fit(zipdata, epochs=epochs)

  return vae


def main():

  def flatImFromDict(adict):
    return tf.squeeze(tf.reshape(tf.cast(adict['image'], 'float32'), (-1, 4096)))

  trainDataRaw = tfds.load("dsprites", split="train")
  trainData = trainDataRaw.map(flatImFromDict)
  trainData = trainData.shuffle(buffer_size=64).batch(64, drop_remainder=True)
  plotData = [tf.cast(adict['image'], 'float32') for i, adict in enumerate(trainDataRaw) if i<5]

  #First test custom, well-controlled loop
  vaeCustom = trainCustom(trainData, epochs=2)
  plotRecons(vaeCustom, plotData, savePlot='recons_custom.png')

  #Next look at build in loops with fit function
  vaeBuiltIn = trainBuiltIn(trainData, epochs=2)
  plotRecons(vaeBuiltIn, plotData, savePlot='recons_built-in.png')


if __name__=="__main__":
  main()