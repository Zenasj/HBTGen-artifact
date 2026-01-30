from tensorflow.keras import layers

import tensorflow.keras as keras
import numpy as np

class MyLoss(keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(self, x, y):
        return x-y

def get_model():
    i = keras.Input((1,))
    m = keras.layers.Dense(10)(i)
    m = keras.layers.Dense(1)(i)
    return keras.Model(i,m)

m = get_model()
m.compile(loss='mse', metrics=[MyLoss()])
m.fit(np.arange(100), np.arange(100))

m.compile(loss='mse', metrics=[MyLoss("myloss")])