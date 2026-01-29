# tf.random.normal((B, 100), dtype=tf.float32) ← The generator input shape is (batch_size, noise_dim=100)

import tensorflow as tf
import numpy as np

class Dense(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Flatten input except batch dimension
        fin = np.prod(input_shape[1:])
        weight_shape = [fin, self.units]

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=0.01)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.matmul(x, self.w)
        return x

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({'units': self.units})
        return config


class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = 0.2
        self.act = tf.keras.layers.LeakyReLU(alpha=self.alpha)

    def call(self, inputs, training=None, mask=None):
        x = self.act(inputs)
        return x

    def get_config(self):
        config = super(LeakyReLU, self).get_config()
        config.update({'alpha': self.alpha})
        return config


class Generator(tf.keras.Model):
    def __init__(self, kernel, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.kernel = kernel

        self.dense0 = Dense(units=7 * 7 * 256)
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.lrelu0 = LeakyReLU()

        self.convt1 = tf.keras.layers.Conv2DTranspose(
            128, self.kernel, strides=(1, 1), padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.lrelu1 = LeakyReLU()

        self.convt2 = tf.keras.layers.Conv2DTranspose(
            64, self.kernel, strides=(2, 2), padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.lrelu2 = LeakyReLU()

        self.convt3 = tf.keras.layers.Conv2DTranspose(
            1, self.kernel, strides=(2, 2), padding='same', use_bias=False,
            activation='tanh')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        z = inputs  # shape (batch_size, noise_dim)

        x = self.dense0(z)
        x = self.bn0(x, training=training)
        x = self.lrelu0(x)

        x = tf.reshape(x, shape=[-1, 7, 7, 256])

        x = self.convt1(x)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)

        x = self.convt2(x)
        x = self.bn2(x, training=training)
        x = self.lrelu2(x)

        x = self.convt3(x)  # Output shape (batch_size, 28, 28, 1)
        return x

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({'kernel': self.kernel})
        return config

    def compute_output_shape(self, input_shape):
        # Workaround to avoid the error expanding shape into multiple outputs:
        # Return a tf.TensorShape to prevent nest from flattening the shape incorrectly
        tf.print('[Generator] - compute_output_shape() input_shape:', input_shape)
        return tf.TensorShape([input_shape[0], 28, 28, 1])

    @tf.function
    def serve(self, z):
        # Separate serving signature method forces training=False for batchnorm etc
        x = self.dense0(z)
        x = self.bn0(x, training=False)
        x = self.lrelu0(x)

        x = tf.reshape(x, shape=[-1, 7, 7, 256])

        x = self.convt1(x)
        x = self.bn1(x, training=False)
        x = self.lrelu1(x)

        x = self.convt2(x)
        x = self.bn2(x, training=False)
        x = self.lrelu2(x)

        x = self.convt3(x)
        return x


class MyModel(tf.keras.Model):
    """
    A wrapper Model encapsulating the original Generator from the issue,
    adjusted with compute_output_shape fix for compatibility with tf.saved_model.save
    and tf.keras.Model.predict.

    Input shape: (batch_size, 100)
    Output shape: (batch_size, 28, 28, 1)
    """

    def __init__(self, kernel=5):
        super(MyModel, self).__init__()
        self.generator = Generator(kernel=kernel)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        # Forward pass delegates to generator call
        return self.generator(inputs, training=training)

    def compute_output_shape(self, input_shape):
        # Delegate to generator's compute_output_shape for consistency
        return self.generator.compute_output_shape(input_shape)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 100], dtype=tf.float32)])
    def serve(self, inputs):
        # Serving signature should call generator's serving method
        return self.generator.serve(inputs)


def my_model_function():
    # Instantiate a MyModel with default kernel=5
    model = MyModel(kernel=5)
    # Build the model by running a dummy input once
    dummy_input = tf.random.normal([1, 100])
    _ = model(dummy_input, training=False)
    return model


def GetInput():
    # Return a random noise tensor compatible with MyModel input shape
    noise_dim = 100
    batch_size = 4  # arbitrarily chosen batch size for example
    return tf.random.normal(shape=(batch_size, noise_dim), dtype=tf.float32)

