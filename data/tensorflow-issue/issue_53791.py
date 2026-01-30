import random
from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Mymodel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.l = tf.keras.models.Sequential([
            layers.Dense(64) for _ in range(3)
        ] + [layers.Dense(1)])
    
    def call(self, inputs, training=None):
        tf.summary.scalar(name='avg_1', data=tf.reduce_sum(inputs))
        return self.l(inputs)

m = Mymodel()
m.compile(loss='mse', optimizer='adam')
m.fit(np.random.randn(1000, 20), np.random.randn(1000), epochs=5)

m.save('/tmp', save_format='tf')