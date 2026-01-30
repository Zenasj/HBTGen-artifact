import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tf.keras.layers as L
def feed_forward_nn():
    with tf.variable_scope("keras_"):
        _input_ = L.Input(shape=(3,))
        layer_1 = L.Dense(5, Z.relu)(_input_)
        layer_2 = L.Dense(5, Z.relu)(layer_1)
        output = L.Dense(1, Z.tanh)(layer_2)
    return Model(inputs=_input_, outputs=output)

model = feed_forward_nn()
sess.run(tf.global_variables_initializer())