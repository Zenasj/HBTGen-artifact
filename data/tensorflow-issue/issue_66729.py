from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import keras

input_shape = [7]
output_shape = [1, 21]

tf_input = keras.Input(input_shape[1:], batch_size=input_shape[0])

tf_z = tf.reshape(tf_input, (1, 7))
tf_output = tf.keras.layers.Dense(21)(tf_z)

model = keras.Model(inputs=[tf_input], outputs=[tf_output])

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

x = Input(...)
...
tf_fn(x)  # Invalid.

class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"