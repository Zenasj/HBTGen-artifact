from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

shape = (1, 1, 1, 10)
# Create a model using high-level tf.keras.* APIs
input = tf.keras.Input(shape=shape[1:], batch_size=1)
initializer = tf.keras.initializers.Constant(value=0)
const = initializer(shape=shape)
output = tf.keras.layers.concatenate([input, const])
model = tf.keras.Model(inputs=[input], outputs=output)
model.summary()
model.compile() # compile the model

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

import tensorflow as tf

shape = (1, 10)
# Create a model using high-level tf.keras.* APIs
input = tf.keras.Input(shape=shape[1:], batch_size=1)
input_2 = tf.keras.Input(shape=shape[1:], batch_size=1)
output = tf.keras.layers.concatenate([input, input_2])
model = tf.keras.Model(inputs=[input, input_2], outputs=output)
model.summary()
model.compile() # compile the model

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

import tensorflow as tf

shape = (1, 10)
# Create a model using high-level tf.keras.* APIs
input = tf.keras.Input(shape=shape[1:], batch_size=1)
initializer = tf.keras.initializers.Constant(value=0)
const = initializer(shape=shape)
output = tf.keras.layers.concatenate([input, const])
model = tf.keras.Model(inputs=[input], outputs=output)
model.summary()
model.compile() # compile the model

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)