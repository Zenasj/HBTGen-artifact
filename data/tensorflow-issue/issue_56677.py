from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test(values):
    vector  = tf.zeros_like(values)
    return vector


class TestLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TestLayer, self).__init__(**kwargs)

    # @tf.function
    def call(self, inputs, training=False):
        tf.print("=============TestLayer.call()============")
        emb_vector = test(values = inputs)
        return emb_vector

class Demo(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Demo, self).__init__(**kwargs)
        
        self.test_layer = TestLayer()        
        self.dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    def call(self, inputs):
        tf.print("=============Demo.call()============")
        vector = self.test_layer(inputs)
        logit = self.dense_layer(vector)
        return logit, vector

    def summary(self):
        inputs = tf.keras.Input(shape=(10,))
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

global_batch_size = 16384
input1 = tf.keras.Input(shape=(10,))
input2 = tf.ones(shape=(global_batch_size,10))

model = Demo()
print("**************Feed TensorSpec*************")
logit1, vector1 = model(input1)

print("**************Feed Tensor*************")
logit2, vector2 = model(input2)