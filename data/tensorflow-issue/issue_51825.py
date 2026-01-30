from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

samples = tf.zeros((1, 256, 256, 3), dtype=tf.float32)
model = Sequential([layers.Conv2D(18, padding='same', kernel_size=3, groups=1), layers.GlobalAveragePooling2D(), layers.Dense(1)])
trouble_model = Sequential([layers.Conv2D(18, padding='same', kernel_size=3, groups=3), layers.GlobalAveragePooling2D(), layers.Dense(1)])

# Backprop on classic model
with tf.GradientTape() as tape:
    out = model(samples, training=True)
grads = tape.gradient(out, model.trainable_weights)

# Now with grouped conv
with tf.GradientTape() as tape:
    out = trouble_model(samples, training=True)
grads = tape.gradient(out, trouble_model.trainable_weights)