from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow.keras.layers as nn
from tensorflow.python.keras.backend import get_graph
from tensorflow.python.keras.utils.generic_utils import to_snake_case
from tensorflow.python.keras.engine.base_layer_utils import unique_layer_name

Module = tf.keras.models.Model 

# class Module(tf.Module):
#     def __init__(self, name=None, **kwargs):
#         super(Module, self).__init__(name='dummy', **kwargs)
#         name = name or to_snake_case(self.__class__.__name__)
#         self._name = unique_layer_name(name, zero_based=True)
#
#     def __call__(self, *args, **kwargs):
#         with get_graph().as_default(), tf.name_scope(self.name):
#             return self.call(*args, **kwargs)

class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.module_list = list(args) if args else []

    def call(self, x):
        for module in self.module_list:
            x = module(x)
        return x

class Block(Module):
    def __init__(self):
        super(Block, self).__init__()
        self.module = Sequential(
                nn.Dense(10),
                nn.Dense(10),)

    def call(self, input_tensor):
        x = self.module(input_tensor)
        return x

class Base(Module):
    def __init__(self):
        super(Base, self).__init__()
        self.module = Sequential(
                Block(),
                Block())

    def call(self, input_tensor):
        x = self.module(input_tensor)
        y = self.module(x)
        return x, y

class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
        self.child = Base()

    def call(self, inputs):
        return self.child(inputs)

net = Network()

inputs = tf.keras.Input(shape=(10, ))
outputs = net(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

print(model.summary(150))