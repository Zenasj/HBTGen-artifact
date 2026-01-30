import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf 

class cm(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs, *args):
        print(f"tf.shape: {tf.shape(inputs)}, inputs.shape:{inputs.shape}")
        return inputs

    @tf.function
    def train_step(self, data):
        print(tf.shape(data), data.shape)
        return {"loss":1}

inputs = tf.random.normal(shape=(100, 10, 1))
c0 = cm()
_ = c0(inputs)
c0.compile("adam")
c0.fit(tf.constant(inputs),epochs=1)

import tensorflow as tf 
import tensorflow_probability.python.distributions as tfd 

class cl(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)
        self.normal = tfd.Normal(loc=[1.] * input_shape[-1], scale=[1.] * input_shape[-1])

    def call(self, inputs, *args):
        batch_shape = inputs.shape[0]
        return self.normal.sample(tf.TensorShape((batch_shape, 2)))


class cm(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.normal = cl()

    def call(self, inputs, *args):
        print(f"tf.shape: {tf.shape(inputs)}, inputs.shape:{inputs.shape}")
        sample = self.normal(inputs)
        return sample

    def train_step(self, data):
        B = tf.shape(data)[0]
        B2 = data.shape[0]
        rnd = keras.backend.random_normal(shape=(B, 2))
        print(f"random normal from keras.backend shape: {rnd.shape}")
        ta = tf.TensorArray(dtype=tf.float32, size=100, element_shape=tf.TensorShape((B2, 1)))
        sample = self(data)  # Error
        return {"loss":1}

inputs = tf.random.normal(shape=(100, 10, 2))
c0 = cm()
c0.compile("adam")
c0.fit(tf.constant(inputs),epochs=1)

c0.fit(tf.constant(inputs),epochs=1, batch_size=10)