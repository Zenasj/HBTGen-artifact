from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

x = tf.constant([[1.0,2.0,3.0], [1.0,2.0,3.0]])
model.predict(x) 

# outputs: array([[0.19964378, 0.23859118, 0.17491119, 0.15458305, 0.23227075],
#       [0.19964378, 0.23859118, 0.17491119, 0.15458305, 0.23227075]],
#      dtype=float32)
# OK!

@tf.function(input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)])
def predict(x):
  return model.predict(x)
p = predict.get_concrete_function()

# TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'

@tf.function(input_signature=[tf.TensorSpec(shape=(3,), dtype=tf.float32)])
def predict(x):
  return model.predict(x)
p = predict.get_concrete_function()

# ValueError: Input 0 of layer dense is incompatible with the layer: 
# expected axis -1 of input shape to have value 3 but received input with shape [None, 1]

@tf.function
def predict(x):
  return model.predict(x)

p = predict.get_concrete_function(x=tf.TensorSpec(shape=(1,3,), dtype=tf.float32))
#  AttributeError: 'Tensor' object has no attribute 'numpy'

# Using Model instead of Model.predict
@tf.function(input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)])
def predict(x):
  return model(x)
p = predict.get_concrete_function()

# Test
x = tf.constant([[1.0,2.0,3.0], [1.0,2.0,3.0]])
p(x) 

# out -> <tf.Tensor: shape=(2, 5), dtype=float32, numpy=
# array([[0.104079  , 0.3223787 , 0.28109816, 0.18215933, 0.11028478],
#       [0.104079  , 0.3223787 , 0.28109816, 0.18215933, 0.11028478]],
#      dtype=float32)>