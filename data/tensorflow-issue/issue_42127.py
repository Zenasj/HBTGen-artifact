import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class TestLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TestLayer, self).__init__()
        self.supports_masking = True

    def call(self, input, mask):
        # simply sum last dimension
        return tf.reduce_sum(input, axis=-1) * tf.cast(mask, tf.float32)
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1]
    
    
layer = tf.keras.layers.TimeDistributed(TestLayer())

# By commenting in the following line, this example works properly
# layer._always_use_reshape = True 

result_ones_mask  = layer(tf.random.normal([1,8,4]), mask=tf.ones ([1,8], dtype=tf.bool))
result_zeros_mask = layer(tf.random.normal([1,8,4]), mask=tf.zeros([1,8], dtype=tf.bool))


print(result_ones_mask.numpy())  # expected 1x8-tensor with random numbers
print(result_zeros_mask.numpy()) # expected 1x8-tensor with zeros

import tensorflow as tf

class ClonedGlobalAveragePooling1D(tf.keras.layers.Layer):
    def __init__(self):
        super(ClonedGlobalAveragePooling1D, self).__init__()        
        self.supports_masking = True
        
    def build(self, input_shape):
        self.l = tf.keras.layers.GlobalAveragePooling1D()
        self.l.build(input_shape)

    def call(self, input, mask=None):
        return self.l(input, mask=mask)
    
    def compute_mask(self, inputs, mask=None):
        return self.l.compute_mask(inputs, mask=mask)
        
    def compute_output_shape(self, input_shape):
        return self.l.compute_output_shape(input_shape)
    
    def compute_output_signature(self, input_shape):
        return self.l.compute_output_signature(input_shape)
        

layer = tf.keras.layers.TimeDistributed(ClonedGlobalAveragePooling1D())   # does NOT work
# layer = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling1D())   # does work

result_ones_mask = layer(tf.random.normal([1,8,4,1]), mask=tf.ones([1,8,4], dtype=tf.bool))
result_zeros_mask = layer(tf.random.normal([1,8,4,1]), mask=tf.zeros([1,8,4], dtype=tf.bool))

print(result_ones_mask.numpy(), result_ones_mask.shape)
print(result_zeros_mask.numpy(), result_zeros_mask.shape)

import tensorflow as tf

class TestLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TestLayer, self).__init__()
        self.supports_masking = True

    def call(self, input, mask=None):
        tf.print("input shape:", tf.shape(input), "; mask:", mask)        
        return input

layer = tf.keras.layers.TimeDistributed(TestLayer())

layer._always_use_reshape = True 

_ = layer(tf.random.normal([2,8,4]), mask=tf.ones ([2,8], dtype=tf.bool))