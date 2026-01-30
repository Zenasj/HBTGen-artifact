from tensorflow import keras
from tensorflow.keras import layers

import time
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self,):
        super(Model, self).__init__()
        self.layers_list = [tf.keras.layers.Dense(2)] + \
            [tf.keras.layers.Dense(1000)] + \
            [tf.keras.layers.Dense(500)]

    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x

model = Model()
model.build((1,2))
weights = model.trainable_variables

input = np.zeros([1,2])

with tf.GradientTape(persistent=True) as tape:
    ts = time.time()
    output = model(input)
    print(f'Forward took {time.time()-ts:.2f}s')

ts = time.time()
gradients = tape.jacobian(output, weights, experimental_use_pfor=False)
print(f'Backward took {time.time()-ts:.2f}s')