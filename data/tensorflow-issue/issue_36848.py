# tf.random.uniform((B, 64), dtype=tf.float32)  ← Input shape: batch size x 64 features

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LeakyReLU

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the custom batch normalization layer used repeatedly
        self.bn1 = CustomBatchNormalization()
        self.dense_1 = Dense(128, activation='linear')
        self.leaky_relu_1 = LeakyReLU(alpha=0.3)

        # Create six layers of CustomBatchNormalization and Dense + LeakyReLU
        self.bn_layers = [CustomBatchNormalization() for _ in range(6)]
        self.dense_layers = [Dense(128, activation='linear') for _ in range(6)]
        self.leaky_relu_layers = [LeakyReLU(alpha=0.3) for _ in range(6)]

        self.bn_out = CustomBatchNormalization()
        self.dense_out = Dense(64, activation='linear')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        x = self.bn1(inputs, training=training)
        x = self.dense_1(x)
        x = self.leaky_relu_1(x)
        for i in range(6):
            x = self.bn_layers[i](x, training=training)
            x = self.dense_layers[i](x)
            x = self.leaky_relu_layers[i](x)
        x = self.bn_out(x, training=training)
        x = self.dense_out(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the model input shape: [batch_size, 64]
    # batch_size is arbitrary; using 4 here for example
    return tf.random.uniform((4, 64), dtype=tf.float32)


class CustomBatchNormalization(layers.Layer):
    def __init__(self, momentum=0.99, epsilon=1e-3,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_range_initializer='ones',
                 **kwargs):
        super(CustomBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_range_initializer = initializers.get(moving_range_initializer)

    def build(self, input_shape):
        dim = input_shape[-1]
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                                     name='gamma',
                                     initializer=self.gamma_initializer,
                                     trainable=True)
        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer=self.beta_initializer,
                                    trainable=True)

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)

        self.moving_range = self.add_weight(
            shape=shape,
            name='moving_range',
            initializer=self.moving_range_initializer,
            trainable=False)

    def call(self, inputs, training=None):
        # To avoid the error "using tf.Tensor as Python bool" in graph mode,
        # test training with tf.cond instead of Python if,
        # and update moving stats through assign not by overwriting variables.

        def training_true():
            mean = tf.reduce_mean(inputs, axis=0)
            maxr = tf.reduce_max(inputs, axis=0)
            minr = tf.reduce_min(inputs, axis=0)
            range_diff = maxr - minr

            # Update moving_mean and moving_range variables by assign
            self.moving_mean.assign(self.moving_mean * self.momentum + mean * (1 - self.momentum))
            self.moving_range.assign(self.moving_range * self.momentum + range_diff * (1 - self.momentum))

            scaled = (inputs - mean) / (range_diff + self.epsilon)
            output = self.gamma * scaled + self.beta
            return output

        def training_false():
            scaled = (inputs - self.moving_mean) / (self.moving_range + self.epsilon)
            output = self.gamma * scaled + self.beta
            return output

        # Use tf.cond to select between training and inference without Python bool
        return tf.cond(tf.cast(training, tf.bool), training_true, training_false) if training is not None else training_true()

    def get_config(self):
        config = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_range_initializer': initializers.serialize(self.moving_range_initializer)
        }
        base_config = super(CustomBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

