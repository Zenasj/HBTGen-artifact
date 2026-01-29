# tf.random.uniform((B, 36, 36, 3), dtype=tf.float64)
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape

# Assumptions & notes:
# - Input images are (36, 36, 3) float64 tensors
# - Latent dimension is 2 per example in the minimal test case
# - Using tfpl.MultivariateNormalTriL and tfpl.IndependentBernoulli probabilistic layers
# - The core issue relates to how the dataset is prepared: inputs must be tuples (x, y) for keras to associate gradients correctly
# - Here we combine encoder and decoder in one tf.keras.Model subclass with full forward pass, enabling clean gradient flow
# - Loss is defined outside as usual and returns expected reconstruction loss from Bernoulli decoding distribution
# - This matches the minimal reproducible example from the issue and the fix (dataset of (x,x) pairs)
# - Using tf.float64 dtype for consistency with the example

latent_dim = 2
input_shape = (36, 36, 3)
kl_regularizer = tfpl.KLDivergenceRegularizer(
    distribution_b=tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)),
    use_exact_kl=False,
    test_points_fn=lambda t: t.sample(3),
    test_points_reduce_axis=None,
    weight=1.0
)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder layers matching the minimal test case
        self.encoder = Sequential([
            Conv2D(3, 3, padding="same", activation=None,  # no activation given in minimal example Conv2D
                   input_shape=input_shape, dtype=tf.float64, name="encoder_conv2d"),
            Flatten(name="encoder_flatten"),
            Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim), name="encoder_dense", dtype=tf.float64),
            tfpl.MultivariateNormalTriL(latent_dim, activity_regularizer=kl_regularizer, dtype=tf.float64, name="encoder_outdist")
        ], name="encoder")

        # Decoder layers from minimal test case
        self.decoder = Sequential([
            Dense(36 * 36 * 3, input_shape=(latent_dim,), dtype=tf.float64, name="decoder_dense"),
            Reshape((36, 36, 3), name="decoder_reshape"),
            Conv2D(3, 3, padding="same", activation=None, dtype=tf.float64, name="decoder_conv2d"),
            Flatten(name="decoder_flatten"),
            tfpl.IndependentBernoulli(event_shape=input_shape, convert_to_tensor_fn=tfd.Bernoulli.logits, name="decoder_outdist")
        ], name="decoder")

    def call(self, inputs, training=False):
        # Forward pass: encode input image to latent dist, sample, decode sample to output dist
        encoded_dist = self.encoder(inputs)
        # Sample from encoded distribution; reparameterization trick handled by tfpl.MultivariateNormalTriL
        z = encoded_dist.sample()
        decoded_dist = self.decoder(z)
        return decoded_dist  # return distribution so loss fn can evaluate log_prob

def my_model_function():
    # Return an initialized instance of MyModel
    return MyModel()

def GetInput():
    # Returns a tuple (x, y) to satisfy keras need for input-label pairs.
    # Both are identical images, as per fix from issue to enable gradient flow.
    # Generate a batch of 20 random images for example.
    batch_size = 20
    # Using float64 as in the original example
    x = tf.random.uniform((batch_size, 36, 36, 3), dtype=tf.float64)
    y = tf.identity(x)  # output is same as input images
    return (x, y)

