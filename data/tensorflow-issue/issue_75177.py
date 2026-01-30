import random
from tensorflow import keras
from tensorflow.keras import layers

# Example 1.

import numpy as np
import tensorflow as tf

np.random.seed(23235)

input_names = ["k", "b", "m", "c", "x"]
input_shapes = [[1, 1], [1, 3], [1, 2], [1, 5], [1, 4]]

inputs = []
outputs = []
for ind in range(len(input_names)):
    input = tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind])
    inputs.append(input)
    outputs.append(tf.keras.layers.Activation(tf.nn.sigmoid)(input))

model_in_memory = tf.keras.Model(inputs=inputs, outputs=outputs)
model_in_memory.export('saved_model')
loaded = tf.saved_model.load('saved_model')
model_from_disk = loaded.signatures['serving_default']

test_list = []
for input_shape in input_shapes:
    test_list.append(np.random.rand(*input_shape))

test_dict = {}
for input_shape, input_name in zip(input_shapes, input_names):
    test_dict[input_name] = np.random.rand(*input_shape)

print('results for original model = ', model_in_memory(test_list))
print('results for saved model = ', model_from_disk(**test_dict))



#Example 2
import numpy as np
import tensorflow as tf

np.random.seed(23235)

input_names = ["k", "b", "m", "c", "x"]
input_shapes = [[1, 1], [1, 3], [1, 2], [1, 5], [1, 4]]

inputs = []
outputs = {}
for ind in range(len(input_names)):
    input = tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind])
    inputs.append(input)
    outputs["name" + str(ind)] = tf.keras.layers.Activation(tf.nn.sigmoid)(input)

model_in_memory = tf.keras.Model(inputs=inputs, outputs=outputs)
model_in_memory.export('saved_model')
loaded = tf.saved_model.load('saved_model')
model_from_disk = loaded.signatures['serving_default']

test_list = []
for input_shape in input_shapes:
    test_list.append(np.random.rand(*input_shape))

test_dict = {}
for input_shape, input_name in zip(input_shapes, input_names):
    test_dict[input_name] = np.random.rand(*input_shape)

print('results for original model = ', model_in_memory(test_list))
print('results for saved model = ', model_from_disk(**test_dict))