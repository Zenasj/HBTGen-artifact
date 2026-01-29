# tf.random.uniform((B, 1), dtype=tf.float32)  # inferred input shape: (batch_size, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A model with a custom layer that adds a custom loss inside call().
    This mimics the minimal example from the issue where a layer
    adds a loss via self.add_loss().
    """

    class CustomLayer(tf.keras.layers.Layer):
        def __init__(self, a):
            super().__init__()
            # Variable initialized from scalar 'a'
            self.var = tf.Variable(a, name='var_a', trainable=True)

        def get_config(self):
            # Adding config for proper serialization support
            config = super().get_config()
            config.update({'a': self.var.numpy()})
            return config

        @classmethod
        def from_config(cls, config):
            # Support deserialization from config dict
            return cls(config['a'])

        def call(self, inputs, training=False):
            # Compute output as sum over last axis: output shape = (batch_size,)
            output = tf.reduce_sum(inputs * self.var, axis=-1)
            # Add a dummy loss as mean of outputs
            self.add_loss(tf.reduce_mean(output))
            return output

    def __init__(self):
        super().__init__()
        self.custom_layer = self.CustomLayer(0.1)

    def call(self, inputs, training=False):
        return self.custom_layer(inputs, training=training)


def my_model_function():
    # Return instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor matching expected shape (batch_size, 1)
    # Using batch_size=4 as a modest default
    batch_size = 4
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)

