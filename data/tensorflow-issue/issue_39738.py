import random
from tensorflow import keras
from tensorflow.keras import layers

from pprint import pprint

import tensorflow as tf

class CustomSequential(tf.keras.Sequential):
    def __init__(self, set_attributes=False, name='CustomSequential', **kwargs):
        super().__init__([
            tf.keras.layers.Conv2D(2, 3, 2, name='conv2'),
            tf.keras.layers.Conv2D(4, 3, 2, name='conv3'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ], name=name, **kwargs)

        if set_attributes:
            self.conv1 = self.layers[0]
            self.conv2 = self.layers[1]
            self.dense = self.layers[3]


class CustomModel(tf.keras.Model):
    def __init__(self, set_attributes, name='CustomModel', **kwargs):
        super().__init__(name=name, **kwargs)
        self.convolution = tf.keras.layers.Conv2D(8, 3, name='conv1')
        self.sequential = CustomSequential(set_attributes, name='MySequential')

    def call(self, inputs):
        net = self.convolution(inputs)

        return self.sequential(net)


print('TensorFlow version')
print(tf.version.GIT_VERSION, tf.version.VERSION)
for set_attributes in [True, False]:
    print(f'Set sequential layer attributes: {set_attributes}')
    inputs = tf.random.uniform([5, 16, 16, 3])
    model = CustomModel(set_attributes)
    outputs = model(inputs)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=model)
    manager = tf.train.CheckpointManager(checkpoint, f'./custom_model_{set_attributes}', max_to_keep=3)
    manager.save()

    pprint(tf.train.list_variables(manager.latest_checkpoint))
    print('------------------')