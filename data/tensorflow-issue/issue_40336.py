# tf.random.uniform((100, 784), dtype=tf.float32) ← Input shape corresponds to batch_size=100 and original_dim=784 flattened MNIST images

import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epsilon_std = 1.0

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder layers
        self.encoder_h = Dense(intermediate_dim, activation='relu')
        self.z_mean_layer = Dense(latent_dim)
        self.z_log_var_layer = Dense(latent_dim)

        # Decoder layers
        self.decoder_h = Dense(intermediate_dim, activation='relu')
        self.decoder_mean = Dense(original_dim, activation='sigmoid')

    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian."""
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
        # z = mean + exp(log_var/2) * epsilon
        return z_mean + tf.exp(z_log_var * 0.5) * epsilon

    def call(self, inputs):
        # Encode input into latent distribution parameters
        h = self.encoder_h(inputs)
        z_mean = self.z_mean_layer(h)
        z_log_var = self.z_log_var_layer(h)

        # Sample latent vector
        z = self.sampling((z_mean, z_log_var))

        # Decode latent vector
        h_decoded = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)

        # Store for loss calculation
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        return x_decoded_mean

class VAE_Loss(Loss):
    def __init__(self, original_dim):
        super().__init__()
        self.original_dim = original_dim

    def call(self, y_true, y_pred):
        # Access encoder outputs stored in model instance via y_pred._keras_history.layer.model
        # Since we cannot pass z_mean and z_log_var directly here, we'll assume they're attached as attributes
        # on the model instance. This requires setting the loss to depend on model internals.
        # Here we assume that y_pred is output of the model call, and the model instance is externally accessible.

        # y_true: original input image
        # y_pred: reconstructed image
        # For this standalone loss function, we need to get z_mean and z_log_var from the model.
        # This needs to be wired externally in model training loop or via a custom training step.
        # For simplification, assume y_pred has attribute 'model' with z_mean and z_log_var, else fallback zero.

        # Since Loss.call does not have direct access to latent variables in this structure,
        # We'll define a function below that combines model + loss properly.

        raise NotImplementedError(
            "VAE_Loss requires access to latent variables; please use a custom training step or compiled loss wrapper."
        )

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor input matching the input expected by MyModel:
    # Shape: (batch_size, original_dim), dtype float32 in range [0, 1]
    return tf.random.uniform((batch_size, original_dim), minval=0., maxval=1., dtype=tf.float32)


# Note:
# The original issue describes an incompatibility of using Keras symbolic tensors inside custom loss with graph mode,
# and a workaround to run eagerly or redesign loss.
#
# To properly implement the VAE loss with latent variables inside a Keras Loss class,
# you either:
# - Use a subclassed model with a custom train_step method that calculates the loss using stored z_mean and z_log_var
# - Use `run_eagerly=True` flag in model.compile
#
# Since the task is to provide a single model and input to match, we reconstruct only the model, sampling, and input here.
#
# For production usage, the custom training loop or custom train_step should be implemented accordingly.

