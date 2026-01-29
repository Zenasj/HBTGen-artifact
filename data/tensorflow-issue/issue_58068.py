# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
import numpy as np

class MyModel(tf.keras.Model):
    """
    Fused Variational Autoencoder model adapted from the given code.
    - Encoder and decoder are encapsulated as submodels.
    - Forward pass runs encoder -> decoder.
    - Custom losses (reconstruction + KL divergence) are implemented separately,
      but not integrated into model.call to keep compatibility with TF 2.20.0 XLA compilation.
    """

    def __init__(self,
                 input_shape=(28, 28, 1),
                 conv_filters=(32, 64, 64, 64),
                 conv_kernels=(3, 3, 3, 3),
                 conv_strides=(1, 2, 2, 1),
                 latent_space_dim=2,
                 reconstruction_loss_weight=1000):
        super().__init__()
        self.input_shape_ = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None

        # Build encoder and decoder submodels
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def call(self, inputs, training=False):
        """
        Forward pass: encoder -> decoder
        Returns reconstructed output.
        """
        latent = self.encoder(inputs, training=training)
        reconstructed = self.decoder(latent, training=training)
        return reconstructed

    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape_, name="encoder_input")
        x = encoder_input
        # Convolutional blocks
        for i in range(self._num_conv_layers):
            x = Conv2D(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernels[i],
                strides=self.conv_strides[i],
                padding="same",
                name=f"encoder_conv_layer_{i+1}"
            )(x)
            x = ReLU(name=f"encoder_relu_{i+1}")(x)
            x = BatchNormalization(name=f"encoder_bn_{i+1}")(x)

        self._shape_before_bottleneck = K.int_shape(x)[1:]  # e.g. (H, W, C)
        x = Flatten(name="encoder_flatten")(x)

        # Dense layers for mean and log variance
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        # Sampling via reparameterization trick
        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            z = mu + K.exp(log_variance / 2) * epsilon
            return z

        encoder_output = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])
        return Model(encoder_input, encoder_output, name="encoder")

    def _build_decoder(self):
        decoder_input = Input(shape=(self.latent_space_dim,), name="decoder_input")
        # Dense layer, reshape to shape before bottleneck
        num_neurons = np.prod(self._shape_before_bottleneck)
        x = Dense(num_neurons, name="decoder_dense")(decoder_input)
        x = Reshape(self._shape_before_bottleneck, name="decoder_reshape")(x)

        # Conv2DTranspose layers in reverse order of encoder conv layers (except first)
        for idx in reversed(range(1, self._num_conv_layers)):
            x = Conv2DTranspose(
                filters=self.conv_filters[idx],
                kernel_size=self.conv_kernels[idx],
                strides=self.conv_strides[idx],
                padding="same",
                name=f"decoder_conv_transpose_layer_{self._num_conv_layers - idx}"
            )(x)
            x = ReLU(name=f"decoder_relu_{self._num_conv_layers - idx}")(x)
            x = BatchNormalization(name=f"decoder_bn_{self._num_conv_layers - idx}")(x)

        # Final Conv2DTranspose layer to reconstruct the image
        x = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )(x)
        decoder_output = Activation("sigmoid", name="sigmoid_output")(x)

        return Model(decoder_input, decoder_output, name="decoder")

    def compute_loss(self, y_true, y_pred):
        """
        Compute the combined VAE loss: weighted reconstruction loss + KL divergence
        y_true: ground truth images
        y_pred: reconstructed images
        Returns: scalar loss tensor (mean over batch)
        """
        # Reconstruction loss: mean squared error per batch element
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
        # KL divergence term referencing encoder layers mu and log_variance
        kl_loss = -0.5 * tf.reduce_sum(1 + self.log_variance - tf.square(self.mu) - tf.exp(self.log_variance), axis=1)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        # Mean over batch
        return tf.reduce_mean(combined_loss)

def my_model_function():
    """
    Create and return an instance of MyModel with default recommended parameters.
    """
    model = MyModel(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2,
        reconstruction_loss_weight=1000,
    )
    return model

def GetInput():
    """
    Return a random tensor simulating a batch of grayscale 28x28 images with 1 channel.
    Shape: (batch_size=4, height=28, width=28, channels=1)
    Values uniformly sampled in [0,1).
    """
    return tf.random.uniform((4, 28, 28, 1), dtype=tf.float32)

