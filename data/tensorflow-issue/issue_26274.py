import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow import keras

X_train = np.random.randn(1000, 8)
y_train = np.random.rand(1000, 1)

class ReconstructingRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(30, activation="elu")
        self.out = keras.layers.Dense(output_dim)

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)

    def call(self, inputs):
        Z = self.hidden(inputs)
        reconstruction = self.reconstruct(Z)
        reconstruction_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.1 * reconstruction_loss)
        return self.out(Z)

model = ReconstructingRegressor(1)
#model.build(tf.TensorShape([None, 8])) # <= works if I add this line
model.compile(loss="mse", optimizer="nadam")
history = model.fit(X_train, y_train, epochs=2) # <= AttributeError (see below)