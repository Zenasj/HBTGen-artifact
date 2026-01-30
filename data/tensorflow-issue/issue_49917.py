import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

def _h(model,x):
    model(x)

class b(tf.keras.layers.Layer):
    def call(self, inputs):
        if tf.constant(True):
            tf.print(tf.random.uniform([5,]),summarize=-1)
        else:
            pass
        return inputs

tf.random.set_seed(123)
inputs = tf.keras.Input(shape=(784,))

x=b()(inputs)
x=b()(x)
x=b()(x)
outputs=b()(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

h=tf.function(_h)

print('first run:')
h(model,tf.constant(np.random.rand(64,784)))
print('second run:')
h(model,tf.constant(np.random.rand(64,784)))

import tensorflow as tf
import numpy as np


def _h(model,x):
    model(x)

class buggy_random(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.boolean_var=tf.Variable(True)
        super().__init__(**kwargs)
    def call(self, inputs):
        if self.boolean_var:
          tf.print(tf.random.uniform([5,]))

        return inputs

tf.random.set_seed(123)
inputs = tf.keras.Input(shape=(784,))

x=buggy_random()(inputs)
x=buggy_random()(x)
x=buggy_random()(x)
outputs=buggy_random()(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.layers[1].boolean_var.assign(False)
model.layers[3].boolean_var.assign(False)


h=tf.function(_h)
print('first run:')
h(model,tf.constant(np.random.rand(64,784)))
print('second run:')
h(model,tf.constant(np.random.rand(64,784)))
'''
first run:
[0.277247906 0.994074821 0.379808426 0.71479249 0.50061965]
[0.277247906 0.994074821 0.379808426 0.71479249 0.50061965]
second run:
[0.176383495 0.109812617 0.334476113 0.66576612 0.116794825]
[0.176383495 0.109812617 0.334476113 0.66576612 0.116794825]
'''

import tensorflow as tf
import numpy as np

def _h(model,x):
    model(x)

class buggy_random(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        for i in tf.range(tf.constant(1)):
          tf.print(tf.random.uniform([5,]))

        return inputs

tf.random.set_seed(123)
inputs = tf.keras.Input(shape=(784,))

x=buggy_random()(inputs)
x=buggy_random()(x)
x=buggy_random()(x)
outputs=buggy_random()(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

h=tf.function(_h)
print('first run:')
h(model,tf.constant(np.random.rand(64,784)))
print('second run:')
h(model,tf.constant(np.random.rand(64,784)))

'''
first run:
[0.277247906 0.994074821 0.379808426 0.71479249 0.50061965]
[0.277247906 0.994074821 0.379808426 0.71479249 0.50061965]
[0.277247906 0.994074821 0.379808426 0.71479249 0.50061965]
[0.277247906 0.994074821 0.379808426 0.71479249 0.50061965]
second run:
[0.176383495 0.109812617 0.334476113 0.66576612 0.116794825]
[0.176383495 0.109812617 0.334476113 0.66576612 0.116794825]
[0.176383495 0.109812617 0.334476113 0.66576612 0.116794825]
[0.176383495 0.109812617 0.334476113 0.66576612 0.116794825]
'''

import tensorflow as tf
import numpy as np

def _h(model,x):
    model(x)

class fixed_random(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.g = tf.random.Generator.from_seed(np.random.randint(2147483647))
        super().__init__(**kwargs)
    def call(self, inputs):
        for i in tf.range(tf.constant(1)):
            tf.print(self.g.uniform([5,]),summarize=-1)
        return inputs

tf.random.set_seed(123)
np.random.seed(88883)
inputs = tf.keras.Input(shape=(784,))

x=fixed_random()(inputs)
x=fixed_random()(x)
x=fixed_random()(x)
outputs=fixed_random()(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

h=tf.function(_h)

print('first run')
h(model,tf.constant(np.random.rand(64,784)))
print('second run')
h(model,tf.constant(np.random.rand(64,784)))
'''
first run
[0.898076177 0.814805269 0.685280442 0.624640703 0.637957335]
[0.438643932 0.923692703 0.202350736 0.259410977 0.551186323]
[0.137919307 0.688777804 0.0962672234 0.532677054 0.639819145]
[0.533203602 0.589777708 0.596117854 0.147099137 0.760685086]
second run
[0.391331553 0.65578413 0.0532933474 0.92483747 0.198105812]
[0.0157146454 0.148599625 0.300893545 0.211638093 0.460565567]
[0.310432553 0.377720356 0.230332017 0.0171860456 0.905866146]
[0.927248478 0.930060267 0.906018138 0.391339064 0.362027764]
'''