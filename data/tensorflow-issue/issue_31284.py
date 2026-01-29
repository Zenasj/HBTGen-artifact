# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST images with batch size 100

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, latent_dim=50):
        super(MyModel, self).__init__()
        self.latent_dim = latent_dim

        # Encoder (inference network)
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),  # Outputs mean and logvar
            ]
        )

        # Decoder (generative network)
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    def encode(self, x):
        # Encodes input into mean and log variance of latent Gaussian
        mean_logvar = self.inference_net(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        # Reparameterization trick
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        # Decode latent vector to reconstructed image logits
        logits = self.generative_net(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    def sample(self, eps=None):
        # Sample from latent space and decode to image
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def call(self, x):
        # Forward pass: encode -> reparameterize -> decode (return reconstruction logits)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        logits = self.decode(z)
        return logits


def my_model_function():
    # Return an initialized instance of the VAE model with latent_dim=50.
    return MyModel(latent_dim=50)

def GetInput():
    # Return a random tensor input matching the model input: batch of 100 MNIST-like images 28x28x1 normalized binarized.
    BATCH_SIZE = 100
    # Random uniform data in [0,1], shape matches normalized MNIST images scaled as binarized 0 or 1.
    x = tf.random.uniform((BATCH_SIZE, 28, 28, 1), minval=0., maxval=1., dtype=tf.float32)
    # Binarize like original preprocessing: >=0.5 -> 1, else 0
    x = tf.where(x >= 0.5, 1.0, 0.0)
    return x

