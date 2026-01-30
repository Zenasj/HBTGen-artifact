import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class NewCell(layers.Layer):
    def __init__(self, name=None):
        
        self.state_size = (
            tf.TensorShape([1, 2]),
            (tf.TensorShape([1, 2]), tf.TensorShape([2, 3])) # error cause by this tuple
        )
        self.output_size = (tf.TensorShape([1, 2]), )
        super().__init__(name=name)
        
    def call(self, inputs, state):
        print('everything is safe and sound!')
        return inputs, state
        
rnn = layers.RNN(NewCell())
inputs = tf.convert_to_tensor(np.random.rand(2, 1, 2))
rnn(inputs)