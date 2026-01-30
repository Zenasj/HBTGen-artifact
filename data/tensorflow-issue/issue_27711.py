from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

root = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=3, kernel_size=2, dilation_rate=2)])
initial_output = root.predict(tf.ones((1, 5, 5, 1)))
tf.saved_model.save(root, "/tmp/sm")

import tensorflow as tf


input_data = tf.keras.layers.Input(name='fts_input', shape=(None,None,3), dtype='float32')
inner = tf.keras.layers.Conv2D(filters=3, kernel_size=3, dilation_rate=2)(input_data)
model = tf.keras.models.Model(inputs=[input_data], outputs=inner)

initial_output = model.predict([tf.ones((2, 300, 300, 3))])
tf.saved_model.save(model, './')