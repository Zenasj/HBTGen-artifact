# tf.random.normal((B, latent_dim), dtype=tf.float32) ← The input shape inferred from the example is (batch_size, 4096) for flattened 64x64 images

import tensorflow as tf
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps flattened 64x64 images to a triplet (z_mean, z_log_var, z)."""
    def __init__(self, latent_dim=32, intermediate_dim=64, **kwargs):
        super().__init__(**kwargs)
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
    """Converts latent vector z back into flattened 64x64 images."""
    def __init__(self, original_dim, intermediate_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class MyModel(tf.keras.Model):
    """
    Variational AutoEncoder model combining Encoder and Decoder.
    Implements the call() method to output reconstructed images and 
    adds the KL divergence regularization loss internally.
    """
    def __init__(self, original_dim=4096, intermediate_dim=64, latent_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Compute KL divergence loss and add to model losses
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)

        return reconstructed


def my_model_function():
    """
    Returns a new instance of MyModel with default parameters compatible 
    with 64x64 grayscale flattened inputs (4096 dims).
    """
    return MyModel(original_dim=4096, intermediate_dim=64, latent_dim=32)


def GetInput():
    """
    Return a random input tensor suitable as input to MyModel.
    Shape is (batch_size=64, 4096) corresponding to batches of 64 flattened 64x64 images.
    Values sampled uniformly from [0, 1].
    """
    batch_size = 64
    original_dim = 4096  # 64*64 flattened grayscale image
    # Using uniform distribution [0,1] since original images are normalized floats.
    return tf.random.uniform((batch_size, original_dim), dtype=tf.float32)

