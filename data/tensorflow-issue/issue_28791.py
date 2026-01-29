# tf.random.normal((batch, original_dim), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    """
    Variational Autoencoder (VAE) model fused with its encoder and decoder,
    adapted from the Keras VAE example. Includes reparameterization sampling.
    
    This version avoids usage of Keras backend `random_normal` that is not supported
    by TFLite by using `tf.random.normal` in a tf.function compatible way.
    """
    def __init__(self, original_dim=784, intermediate_dim=64, latent_dim=32):
        super().__init__()
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        # Encoder network
        self.encoder_input = Input(shape=(self.original_dim,), name="encoder_input")
        x = Dense(self.intermediate_dim, activation="relu")(self.encoder_input)
        x = Dense(self.intermediate_dim, activation="relu")(x)
        self.z_mean_layer = Dense(self.latent_dim, name="z_mean")
        self.z_log_var_layer = Dense(self.latent_dim, name="z_log_var")
        self.z_mean = self.z_mean_layer(x)
        self.z_log_var = self.z_log_var_layer(x)

        # Sampling with reparameterization trick using tf.random.normal
        # Use a Lambda layer so it can be traced/compiled with @tf.function
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name="z")([self.z_mean, self.z_log_var])

        # Decoder network
        self.latent_inputs = Input(shape=(self.latent_dim,), name="z_sampling")
        x_dec = Dense(self.intermediate_dim, activation="relu")(self.latent_inputs)
        x_dec = Dense(self.intermediate_dim, activation="relu")(x_dec)
        self.decoder_output_layer = Dense(self.original_dim, activation="sigmoid")
        decoder_output = self.decoder_output_layer(x_dec)

        # Instantiate encoder and decoder models for modularity
        self.encoder = Model(self.encoder_input, [self.z_mean, self.z_log_var, self.z], name="encoder")
        self.decoder = Model(self.latent_inputs, decoder_output, name="decoder")

        # Full VAE model: input -> encoder -> sampled latent -> decoder -> output
        vae_output = self.decoder(self.z)
        self.vae = Model(self.encoder_input, vae_output, name="vae_mlp")

        # Define the VAE loss function
        def vae_loss(x_true, x_pred):
            reconstruction_loss = mse(x_true, x_pred)
            reconstruction_loss *= self.original_dim
            kl_loss = 1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var)
            kl_loss = tf.reduce_sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            return tf.reduce_mean(reconstruction_loss + kl_loss)

        self.vae.add_loss(vae_loss(self.encoder_input, vae_output))
        self.vae.compile(optimizer="adam")  # Use adam optimizer by default

    @staticmethod
    def sampling(args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.
        Args:
            args (tuple): (z_mean, z_log_var)
        Returns:
            sampled latent vector z
        """
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))  # use tf.random.normal directly
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=False):
        """
        Forward pass through the VAE model.
        inputs: tensor with shape (batch, original_dim)
        returns: reconstruction output with shape (batch, original_dim)
        """
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstructed = self.decoder(z, training=training)
        return reconstructed

def my_model_function():
    """
    Returns an instance of MyModel with default dimensions.
    """
    return MyModel()

def GetInput():
    """
    Returns an input tensor compatible with MyModel.
    Assumes original_dim=784 (e.g. flattened 28x28 grayscale images).
    Batch size chosen as 4 for demonstration.
    """
    batch_size = 4
    original_dim = 784
    # Generate a random tensor with values between 0 and 1
    # Shape: (batch_size, original_dim)
    return tf.random.uniform((batch_size, original_dim), minval=0.0, maxval=1.0, dtype=tf.float32)

