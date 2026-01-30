from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

input = tf.keras.layers.Input(shape = [None,None,3])
output = tf.keras.layers.Conv2D(filters=1,kernel_size=[1,1])(input)
model = tf.keras.Model(inputs=input,outputs=output)
# The model consists of a single 1x1 convolution. Therefore the spatial extent of the output is always trivially equal to
# the spatial extent of the input.

# We must start from 257 because, when creating ints between -5 and 256 Python returns references to singletons.
# Therefore there is a one-to-one mapping between int value and object-id in this range and the cache works as
# intended. Outside this range, no such guarantee is possible and the cache fails.
for x in range(257,300):
    shape=model.compute_output_shape([[1,x,x,3]])
    assert (shape[1]==x)