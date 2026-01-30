from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
inputs = tf.keras.layers.Input(shape=(10,))
hiddens = tf.keras.layers.Dense(15,  trainable=True, name="trainable_layer")(inputs)
output = tf.keras.layers.Dense(5, trainable=False, name="nontrainable_layer")(hiddens)
model = tf.keras.Model(inputs, output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
print(f"Model trainable variables: {model.trainable_variables}")
print(f"Model non-trainable variables: {model.non_trainable_variables}")
print(f"Trainable flags on model non-trainable variables: {[v.trainable for v in model.non_trainable_variables]}")