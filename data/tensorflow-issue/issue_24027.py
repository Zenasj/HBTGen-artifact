from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.enable_eager_execution()
tf.executing_eagerly()


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__(name='mymodel')
        
        self.f = tf.keras.layers.Dense(units=10)
        self.g = tf.keras.layers.Dense(units=10)

#         Workaround
#         self.f = tf.keras.Sequential([tf.keras.layers.Dense(units=10)])
#         self.g = tf.keras.Sequential([tf.keras.layers.Dense(units=10)])

    def build(self, input_shapes):
        self.f.build(input_shapes[0])
        self.g.build(input_shapes[1])
        self.built = True
    
    def call(self, x, y):
        return self.f(x) + self.g(y)


model = MyModel()
model.build([(None, 5), (None, 3)])

for v in model.variables:
    print(v.name)