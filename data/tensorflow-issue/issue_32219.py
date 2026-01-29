# tf.random.uniform((B, imsize, imsize, color_channels), dtype=tf.float32)
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers.distribution_layer

# Assumptions based on the issue:
# - Input shape is (batch_size, imsize, imsize, color_channels).
# - latent_dim=8
# - The encoder outputs a tfp.distributions.MultivariateNormalTriL distribution.
# - The decoder outputs a reconstructed image tensor.
# - The model uses a VAE structure.
#
# The problem reported was that calling encoder.predict() fails because the
# output distribution instance is not a tensor and Keras expects tensor-like outputs.
# A common workaround to get the latent code as a tensor is to explicitly call
# `sample()` or `mean()` on the distribution before outputting it in the encoder.
#
# Here, to fix the issue, the encoder will output the latent *samples* (samples drawn
# from the distribution) instead of the distribution object itself.
#
# This approach lets the encoder still represent uncertainty internally,
# but exposes a tensor compatible with Keras predict().
#
# The class MyModel encapsulates both the encoder and decoder, with a forward pass
# returning the decoded reconstructed image for input images.
#
# GetInput() returns a random input tensor consistent with the input shape.

latent_dim = 8
color_channels = 1
imsize = 28  # MNIST images 28x28

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Encoder
        self.flatten = tf.keras.layers.Flatten(name="encoder_flatten")
        self.dense1 = tf.keras.layers.Dense(500, activation='softplus', name="Inference-l1_Dense")
        self.dense_params = tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim),
                                                  name="Inference-l2_Dense_params")
        self.distribution_layer = tfpl.MultivariateNormalTriL(latent_dim, name="encoder_distribution")
        
        # Prior distribution for KLD loss
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1.0),
                                     reinterpreted_batch_ndims=1)
        self.kl_loss_layer = tfpl.KLDivergenceAddLoss(self.prior, weight=1.0)

        # Decoder
        self.decoder_dense1 = tf.keras.layers.Dense(500, activation='softplus', name="Generative-l1_Dense")
        self.decoder_dense2 = tf.keras.layers.Dense(imsize * imsize * color_channels, activation='sigmoid',
                                                    name="Generative-l2_Dense_out")
        self.reshape_layer = tf.keras.layers.Reshape(target_shape=(imsize, imsize, color_channels),
                                                     name="Generative-output_probs")

    def encode(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        params = self.dense_params(x)
        q_z = self.distribution_layer(params)
        # Add KL divergence loss
        _ = self.kl_loss_layer(q_z)
        return q_z

    def decode(self, z):
        x = self.decoder_dense1(z)
        x = self.decoder_dense2(x)
        x = self.reshape_layer(x)
        return x

    def call(self, inputs, training=False):
        # Forward pass through encoder then decoder
        q_z = self.encode(inputs)
        # For training and evaluation, use z sampled from q_z to keep stochasticity
        z = q_z.sample()
        reconstructed = self.decode(z)
        return reconstructed

    def encode_latent(self, inputs):
        # Returns latent samples as tensor for external use (e.g. encoder.predict())
        q_z = self.encode(inputs)
        return q_z.sample()

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random batch of images with shape [batch_size, imsize, imsize, color_channels]
    # Batch size chosen arbitrarily as 4 for testing/predict compatibility.
    batch_size = 4
    return tf.random.uniform((batch_size, imsize, imsize, color_channels),
                             minval=0., maxval=1., dtype=tf.float32)

