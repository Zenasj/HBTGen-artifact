# tf.random.uniform((32, 16), dtype=tf.float32) ← Input shape inferred from original example: batch_size=32, feature_dim=16

import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None, mask=None):
        # Add a metric with a constant tensor of 2.0 values (shape [32])
        # In the original issue, this metric got mixed up with others when added here.
        self.add_metric(tf.ones([tf.shape(inputs)[0]]) * 2.0, name='two', aggregation='mean')
        return inputs


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self._sampler = MyLayer(name='sampler')

    def call(self, inputs, training=None, mask=None):
        x = self._sampler(inputs)
        # Add additional metrics with different constant tensors
        self.add_metric(tf.ones([tf.shape(inputs)[0]]) * 1.0, name='one', aggregation='mean')
        self.add_metric(tf.ones([tf.shape(inputs)[0]]) * 3.0, name='three', aggregation='mean')
        return x


def my_model_function():
    # Return an instance of the custom Model
    return MyModel()


def GetInput():
    # Return a random input tensor matching the expected input shape (batch=32, features=16)
    return tf.random.uniform((32, 16), dtype=tf.float32)

