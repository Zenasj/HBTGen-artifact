from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
for input_shape in [(1,), (1, 1)]:
    print("input_shape", input_shape)
    
    # Code Section 1
    sub_in = tf.keras.Input((1,))
    relu_layer = tf.keras.layers.ReLU()
    sub_out = relu_layer(sub_in)
    submodel = tf.keras.Model(sub_in, sub_out)

    assert len(relu_layer.inbound_nodes) == 1

    # Code section 2
    inp = tf.keras.Input(input_shape)
    out = submodel(inp)

    assert len(relu_layer.inbound_nodes) == 2

import tensorflow as tf
for input_shape in [(1,), (1, 1)]:
    print("input_shape", input_shape)
    
    # Code Section 1
    sub_in = tf.keras.Input((1,))
    relu_layer = tf.keras.layers.ReLU()
    sub_out = relu_layer(sub_in)
    submodel = tf.keras.Model(sub_in, sub_out)

    assert len(relu_layer.inbound_nodes) == 1

    # Code section 2
    inp = tf.keras.Input(input_shape)
    out = relu_layer(inp)

    assert len(relu_layer.inbound_nodes) == 2