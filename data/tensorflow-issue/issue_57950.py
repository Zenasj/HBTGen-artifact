# tf.random.uniform((B, 80, 24, 1), dtype=tf.float32) ← Assumed input shape from data generator reshape

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, hp=None):
        super().__init__()
        # Hyperparameters can be passed in or default values used
        # If hp (kerastuner.HyperParameters) is None, use fixed defaults
        init = tf.keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform')

        if hp is None:
            # Default parameters for the layers
            num_filters1 = 8
            num_filters2 = 32
            num_filters3 = 48
            num_filters4 = 48
            num_dense = 200
            latent_space_dim = 20
        else:
            num_filters1 = hp.Choice("num_filters_layer_1", values=[8, 32], default=8)
            num_filters2 = hp.Int("num_filters_layer_2", min_value=16, max_value=64, step=16)
            num_filters3 = hp.Int("num_filters_layer_3", min_value=16, max_value=96, step=16)
            num_filters4 = hp.Int("num_filters_layer_4", min_value=16, max_value=112, step=16)
            num_dense = hp.Int("num_Dense", 0, 600, 200)
            latent_space_dim = hp.Int("latent_space", 20, 40, 10)

        # Encoder (CNN)
        self.conv1 = layers.Conv2D(filters=num_filters1, kernel_size=3,
                                   activation='relu', kernel_initializer=init,
                                   padding='same', strides=1)
        self.conv2 = layers.Conv2D(filters=num_filters2, kernel_size=3,
                                   activation='relu', kernel_initializer=init,
                                   padding='same', strides=1)
        self.conv3 = layers.Conv2D(filters=num_filters3, kernel_size=3,
                                   activation='relu', kernel_initializer=init,
                                   padding='same', strides=1)
        self.conv4 = layers.Conv2D(filters=num_filters4, kernel_size=3,
                                   activation='relu', kernel_initializer=init,
                                   padding='same', strides=1)

        self.flatten = layers.Flatten()
        self.dense_encoding = layers.Dense(num_dense, activation='relu', name="encoding_5")
        self.latent_dense = layers.Dense(latent_space_dim, activation='relu')

        # Decoder (CNN)
        self.decoder_dense1 = layers.Dense(num_dense, activation='relu')
        # Need to determine shape after flatten to reshape correctly;
        # We'll fix it in build() method after seeing input shape.
        self.decoder_dense2 = None  # placeholder, created in build()
        self.reshape_layer = None  # placeholder, created in build()

        self.deconv3 = layers.Conv2DTranspose(filters=num_filters3, kernel_size=3,
                                              activation='relu', kernel_initializer=init,
                                              padding='same', strides=1)
        self.deconv4 = layers.Conv2DTranspose(filters=num_filters2, kernel_size=3,
                                              activation='relu', kernel_initializer=init,
                                              padding='same', strides=1)
        self.deconv5 = layers.Conv2DTranspose(filters=num_filters1, kernel_size=3,
                                              activation='relu', kernel_initializer=init,
                                              padding='same', strides=1)
        self.deconv6 = layers.Conv2DTranspose(filters=1, kernel_size=3,
                                              activation='sigmoid', kernel_initializer=init,
                                              padding='same', strides=1)

        # We track some variables to help rebuilding shapes on build
        self._layer3_shape = None
        self._layer_flatten_dim = None

    def build(self, input_shape):
        # input_shape expected: (batch_size, 80, 24, 1)
        x = tf.random.uniform(input_shape)
        # Compute shape after conv4 layers to build decoder reshape layer correctly
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        self._layer3_shape = x.shape[1:]  # (80, 24, filters)
        self._layer_flatten_dim = tf.keras.layers.Flatten().compute_output_shape(x.shape)[-1]

        self.decoder_dense2 = layers.Dense(self._layer_flatten_dim, activation='relu')
        self.reshape_layer = layers.Reshape(self._layer3_shape)

        super().build(input_shape)

    def call(self, inputs, training=False):
        # Encoder
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x_flat = self.flatten(x)
        h = self.dense_encoding(x_flat)
        latent = self.latent_dense(h)

        # Decoder
        x = self.decoder_dense1(latent)
        x = self.decoder_dense2(x)
        x = self.reshape_layer(x)

        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        output = self.deconv6(x)
        return output

def my_model_function():
    # Return an instance of MyModel with no tuner hyperparameters (default values)
    return MyModel()

def GetInput():
    # Generate a random tensor input consistent with model input shape:
    # batch size 1, height 80, width 24, channels 1, float32
    return tf.random.uniform((1, 80, 24, 1), dtype=tf.float32)

