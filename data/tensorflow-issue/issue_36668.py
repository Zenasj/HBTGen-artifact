import random
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras import layers

class InnerLayer(layers.Layer):

    def __init__(self):
        super(InnerLayer, self).__init__()

    def call(self, inputs):
        # Should always print 1.0
        self.add_metric(tf.constant(1.0), aggregation='mean',
                        name='should_be_1')
        return inputs


class OuterModel(tf.keras.Model):
    def __init__(self):
        super(OuterModel, self).__init__()
        self.inner = InnerLayer()

    def call(self, inputs):
        # Should always print 2.0
        self.add_metric(tf.constant(2.0), aggregation='mean',
                        name='should_be_2')
        return self.inner(inputs)


model = OuterModel()
model.compile('adam', loss='mse')
dataset = tf.random.normal((100, 5))
model.fit(dataset, dataset, epochs=10, steps_per_epoch=100, verbose=2)