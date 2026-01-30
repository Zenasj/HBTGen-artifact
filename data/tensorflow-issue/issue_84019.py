import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MockLayer(tf.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = tf.keras.Variable(tf.random.normal([5, 5]), name="m")
        self.w = tf.keras.Variable(tf.random.normal([5, 5]), name="w")

    def __call__(self, inputs):
        return self.m * inputs

layer1 = MockLayer()
print([v.name for v in layer1.trainable_variables])

class MockLayer(tf.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = tf.Variable(tf.random.normal([5, 5]), name="m")
        self.w = tf.Variable(tf.random.normal([5, 5]), name="w")

    def __call__(self, inputs):
        return self.m * inputs

layer1 = MockLayer()
print([v.name for v in layer1.trainable_variables])

class MockLayer(tf.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = tf.keras.layers.LayerNormalization(*args, **kwargs)

    def __call__(self, inputs):
        return self.norm(inputs)