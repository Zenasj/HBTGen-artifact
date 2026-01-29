# tf.random.uniform((B, 4), dtype=tf.float32)  ← Input shape is (batch_size, 4)

import tensorflow as tf
from tensorflow.keras import layers, initializers

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

        # Initializers for weights and moving stats
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_range_initializer = initializers.get(moving_range_initializer)

    def build(self, input_shape):
        dim = input_shape[-1]
        shape = (dim,)

        self.gamma = self.add_weight(
            shape=shape,
            name='gamma',
            initializer=self.gamma_initializer,
            trainable=True
        )
        self.beta = self.add_weight(
            shape=shape,
            name='beta',
            initializer=self.beta_initializer,
            trainable=True
        )

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False
        )

        self.moving_range = self.add_weight(
            shape=shape,
            name='moving_range',
            initializer=self.moving_range_initializer,
            trainable=False
        )
        super(CustomBatchNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        # Use the "training" argument to switch behavior

        if training is False:
            # Inference mode: use moving mean and moving_range
            scaled = (inputs - self.moving_mean) / (self.moving_range + self.epsilon)
            return self.gamma * scaled + self.beta

        # Training mode: compute batch statistics
        mean = tf.math.reduce_mean(inputs, axis=0)
        maxr = tf.math.reduce_max(inputs, axis=0)
        minr = tf.math.reduce_min(inputs, axis=0)

        range_diff = maxr - minr

        # Update moving averages
        self.moving_mean.assign(self.moving_mean * self.momentum + (1 - self.momentum) * mean)
        self.moving_range.assign(self.moving_range * self.momentum + (1 - self.momentum) * range_diff)

        scaled = (inputs - mean) / (range_diff + self.epsilon)
        return self.gamma * scaled + self.beta

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


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Create layers matching the model described
        self.custom_bn1 = CustomBatchNormalization()
        self.dense1 = layers.Dense(24, activation='linear')
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.3)

        self.custom_bn2 = CustomBatchNormalization()
        self.dense2 = layers.Dense(128, activation='linear')
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.3)

        self.custom_bn3 = CustomBatchNormalization()
        self.out_dense = layers.Dense(5, activation='linear')

    def call(self, inputs, training=None):
        x = self.custom_bn1(inputs, training=training)
        x = self.dense1(x)
        x = self.leaky_relu1(x)
        x = self.custom_bn2(x, training=training)
        x = self.dense2(x)
        x = self.leaky_relu2(x)
        x = self.custom_bn3(x, training=training)
        outputs = self.out_dense(x)
        return outputs


def my_model_function():
    model = MyModel()
    # Build the model with a dummy input to create weights (optional)
    dummy_input = tf.zeros((1, 4))
    model(dummy_input, training=False)  # build call to create variables
    return model


def GetInput():
    # Returns a random input tensor compatible with MyModel input shape (batch, 4)
    # Assumption: batch size 4 for example
    return tf.random.uniform((4, 4), dtype=tf.float32)

