from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
input = tf.keras.Input(
            shape=(10,1),
)
output = tf.keras.layers.LocallyConnected1D(
            1,
            5,
            5,
            implementation=3,  # implementation=1 does not cause error
)(input)
model = tf.keras.Model(
            inputs=input,
            outputs=output,
)
tf.saved_model.save(model, "tmp")