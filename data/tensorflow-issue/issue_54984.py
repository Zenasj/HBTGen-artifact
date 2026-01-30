import math
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.layer = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.layer(inputs)

model = MyModel()
x = tf.zeros((1, 1))
with tf.GradientTape() as tape:
    theta = model(x)[0,0]
    loss = tf.concat([tf.math.cos(theta), tf.math.sin(theta)], axis=0)
grads = tape.gradient(loss, model.trainable_variables)

from typing import Union, List, Mapping, Any
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, hidden_units:List[int]):
        super(MyModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [tf.keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self) -> Mapping[str,Any]:
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config:Mapping[str,Any]) -> 'MyModel':
        return cls(**config)

model = MyModel([3,2])
model.from_config(model.get_config())
x = tf.zeros((1, 1))
print(x)
with tf.GradientTape() as tape:
    theta = model(x)[0]
    loss = tf.concat([tf.math.cos(theta), tf.math.sin(theta)], axis=0)
grads = tape.gradient(loss, model.trainable_variables)

from typing import Union, List, Mapping, Any
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, hidden_units:List[int]):
        super(MyModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [tf.keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self) -> Mapping[str,Any]:
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config:Mapping[str,Any]) -> 'MyModel':
        return cls(**config)

model = MyModel([3,2])
model.from_config(model.get_config())
x = tf.zeros((1, 1))
print(x)
with tf.GradientTape() as tape:
    theta = model(x)[0,0]
    loss = tf.concat([tf.math.cos(theta), tf.math.sin(theta)], axis=0)
grads = tape.gradient(loss, model.trainable_variables)