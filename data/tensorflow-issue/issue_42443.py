from tensorflow import keras

import tensorflow as tf

# Define a simple model with named input
a = tf.keras.Input(shape=[1], name='a')
b = tf.keras.Input(shape=[1], name='b')
out = tf.concat([a, b], axis=0)
out = tf.reduce_sum(out, axis=0)
model = tf.keras.Model([a, b], [out])

# As expected models runs with named input
model({'a': tf.convert_to_tensor([1]), 'b': tf.convert_to_tensor([1])})

# However, model will silently accept any dictionary with two tensors
# (of appropriate shape) without warning.
model({'c': tf.convert_to_tensor([1]), 'd': tf.convert_to_tensor([1])})

# It will warn if the dictionary has too many entries
model({'c': tf.convert_to_tensor([1]), 'd': tf.convert_to_tensor([1]), 'e': tf.convert_to_tensor([1])})

# Or raise an Assertion if the dictionary has too few entries.
model({'c': tf.convert_to_tensor([1])})

# Or raise an Assertion if the dictionary has too few entries.
model({'c': tf.convert_to_tensor([1])})

# However, model will silently accept any dictionary with two tensors
# (of appropriate shape) without warning.
model({'c': tf.convert_to_tensor([1]), 'd': tf.convert_to_tensor([1])})

# In the case that the graph is constructed with dict input tensors,
        # We will use the original dict key to map with the keys in the input
        # data. Note that the model.inputs is using nest.flatten to process the
        # input tensors, which means the dict input tensors are ordered by their
        # keys.