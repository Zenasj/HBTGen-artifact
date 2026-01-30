import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

X = np.random.rand(100, 30)
Y_1 = np.random.rand(100, 1)
Y_2 = np.random.rand(100, 5)


# FUNCTIONAL API is ok

inputs = tf.keras.layers.Input((30,))

output_1 = tf.keras.layers.Dense(1, name="myname1")(inputs)
output_2 = tf.keras.layers.Dense(5, name="myname2")(inputs)


model = tf.keras.Model(inputs=inputs, outputs=[output_1, output_2])

losses = {
    "myname1": 'mse',
    "myname2": 'mse',
}

model.compile(optimizer='adam', loss=losses)

model.fit(x=X, y=(Y_1, Y_2), epochs=2)

# SUBCLASSING raises error

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(1, name="myname1")
        self.dense2 = tf.keras.layers.Dense(5, name="myname2")
        
    def call(self, x):
        return self.dense1(x), self.dense2(x) 


model = MyModel()

losses = {
    "myname1": 'mse',
    "myname2": 'mse',
}

model.compile(optimizer='adam', loss=losses)

model.fit(x=X, y=(Y_1, Y_2), epochs=2)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(1, name="myname1")
        self.dense2 = tf.keras.layers.Dense(5, name="myname2")
        
    def call(self, x):
        return ({"myname1": self.dense1(x), "myname2": self.dense2(x)}, {"myname3": self.dense1(x), "myname4": self.dense2(x)})


model = MyModel()

losses = {
    "myname1": 'mse',
    "myname2": 'mse',
}

metrics = (
    {
    "myname1": ['mse'],
    "myname2": ['mse'],
    },
    {
    "myname3": ['mse'],
    "myname4": ['mse'],
    },
)
model.compile(optimizer='adam', loss=(losses, None) , metrics=metrics)


X = np.random.rand(100, 30)
Y_1 = np.random.rand(100, 1)
Y_2 = np.random.rand(100, 5)

model.fit(x=X, y=({"myname1": Y_1, "myname2": Y_2}, {"myname3": Y_1, "myname4": Y_2}), epochs=2)

model.fit(x=X, y={"myname1": Y_1, "myname2": Y_2}, epochs=2)