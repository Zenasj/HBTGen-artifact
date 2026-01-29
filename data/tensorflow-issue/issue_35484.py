# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape inferred as batch size x original_dim (784 flattened MNIST image)

import tensorflow as tf
from tensorflow.keras import layers

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
        # inputs shape: (batch_size, original_dim)
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


class MyModel(tf.keras.Model):
    """
    Variational Autoencoder model that combines Encoder and Decoder.
    Fixed to support latent_dim=1 by explicitly building encoder layers with fixed input shape,
    avoiding Dense layer issues with undefined last input dimension.
    """

    def __init__(self,
                 original_dim=784,
                 intermediate_dim=64,
                 latent_dim=1,
                 name='autoencoder',
                 **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim

        # Define encoder and decoder instances
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim,
                               intermediate_dim=intermediate_dim)

        # Fix for latent_dim=1 issue:
        # Build layers with known input shapes by calling them once with dummy data
        self._built = False
        self._latent_dim = latent_dim

    def build(self, input_shape):
        # Build encoder and decoder explicitly once shapes are known
        if not self._built:
            # Build encoder layers sequentially
            dummy_input = tf.zeros(input_shape)
            self.encoder.dense_proj(dummy_input)  # build dense_proj with defined input
            proj_out_shape = self.encoder.dense_proj.output_shape

            # Build dense_mean and dense_log_var with fixed input_shape (proj_out_shape)
            self.encoder.dense_mean.build(proj_out_shape)
            self.encoder.dense_log_var.build(proj_out_shape)

            # Build decoder layers
            # Decoder input_shape is (batch_size, latent_dim)
            dummy_latent = tf.zeros((input_shape[0], self._latent_dim))
            self.decoder.dense_proj(dummy_latent)
            self.decoder.dense_output.build(self.decoder.dense_proj.compute_output_shape(dummy_latent.shape))

            self._built = True
        super(MyModel, self).build(input_shape)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed


def my_model_function():
    # Returns an instance of MyModel configured like the original VAE with latent_dim=1
    return MyModel(original_dim=784, intermediate_dim=64, latent_dim=1)


def GetInput():
    # Returns a random input tensor matching expected input: batch_size x original_dim (784)
    # Assumption: float32 inputs in [0,1], typical for normalized MNIST data
    batch_size = 32  # arbitrary batch size
    original_dim = 784
    # Create random uniform floats to simulate normalized image data
    return tf.random.uniform((batch_size, original_dim), minval=0.0, maxval=1.0, dtype=tf.float32)

