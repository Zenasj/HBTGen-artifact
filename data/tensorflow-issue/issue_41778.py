from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=[10, 5, 1])
x = inputs
x = Conv2D(32, (3, 3))(x) # *** Functions correctly when removed
x = Flatten()(x)
x = Dense(1)(x)
    
model = Model(inputs, x)
model.compile()

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile("min_model.tflite", "wb") as f:
    f.write(tflite_model)