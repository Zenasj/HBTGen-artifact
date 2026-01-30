import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
layers = tf.keras.layers
keras = tf.keras

# From documentation until the next comment
class Linear(layers.Layer):

    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config
    

layer = Linear(10)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)

# Creating a layer and saving its weights
data = np.random.random((1000, 10))
labels = np.random.random((1000, 10))
inputs = keras.Input((10,))
outputs = layer(inputs)
model = keras.Model(inputs, outputs)
config = model.get_config()
print(config)
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=10, epochs=1)

model.save_weights("temp/layers_weights")

import numpy as np
import tensorflow as tf
layers = tf.keras.layers
keras = tf.keras

# From documentation until the next comment
class Linear(layers.Layer):

    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config
    

layer = Linear(10)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)

# Creating a layer and saving its weights
data = np.random.random((1000, 10))
labels = np.random.random((1000, 10))
inputs = keras.Input((10,))
outputs = layer(inputs)
model = keras.Model(inputs, outputs)
config = model.get_config()
print(config)
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=10, epochs=2)
model.save_weights("temp/layers_weights_inh5format",save_format="h5")