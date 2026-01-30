from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import numba
import tensorflow as tf

@numba.jit(nopython = True)
def func(param, input):
    return param*input**2

@numba.jit(nopython = True)
def gradfunc(param, input):
    return input**2

@tf.custom_gradient
def func_tf(param, input):
    def grad(dy):
        return tf.numpy_function(gradfunc, (param.numpy(), input.numpy()), tf.float32), 2*param*input
    return tf.numpy_function(func, (param.numpy(), input.numpy()), tf.float32), grad

class myLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def build(self, input_shape):
        self.param = self.add_weight("param")
        
    def call(self, input):
        return func_tf(self.param, input)
    
class myModel(tf.keras.Model):
    def __init__(self, num_layers):
        super().__init__(name='')
        self._layers = [myLayer() for _ in range(num_layers)]
        
    def call(self, input_tensor):
        for layer in self._layers:
            input_tensor = layer(input_tensor)
        return input_tensor
    
model = myModel(3)
print(model(1.5)) # <-- this works

def loss(target, output):
    tf.abs(tf.reduce_sum(target - output))**2

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=loss,
    metrics=[loss])

model.fit([0.1], [0.4], batch_size=None)