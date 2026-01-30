import random

from tensorflow.keras import layers, Model, remat
import numpy as np
class CustomRematLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remat_function = remat(self.intermediate_function)
    def intermediate_function(self, x):
        return x * 1.0 
    def call(self, inputs):
        return self.remat_function(inputs)

inputs = layers.Input(shape=(0,)) 
x = CustomRematLayer()(inputs) 
outputs = layers.Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="sgd", loss="mse")
model.predict(np.random.randn(32, 0))