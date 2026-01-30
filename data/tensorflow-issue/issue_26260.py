import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
import numpy as np

training_set_size = 32 * 10 # <= works only if self.add_metric() is added and this is a multiple of 32
X = np.random.randn(training_set_size, 8) 
y = np.random.randn(training_set_size, 1)

class MyModel(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.out = keras.layers.Dense(output_dim)
        self.my_metric = keras.metrics.Mean()

    def call(self, inputs):
        #self.add_metric(self.my_metric(5.)) # <= Works if you add this line, but model is stateful
        return self.out(inputs)

model = MyModel(1)
model.compile(loss="mse", optimizer="nadam")
history = model.fit(X, y, epochs=2)