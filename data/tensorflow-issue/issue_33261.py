from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

time_distributed_layer = tf.keras.layers.TimeDistributed(
    layer=tf.keras.layers.Conv2D(filters=16, kernel_size=3),
    input_shape=(None, 100, 100, 3),
)
model = tf.keras.Sequential([time_distributed_layer])
model.save('model')  # Works

# Check correct restoration
restored_model = tf.keras.models.load_model('model')
for weight, restored_weight in zip(model.weights, restored_model.weights):
    assert weight.name == restored_weight.name
    tf.debugging.assert_equal(weight, restored_weight)

import tensorflow as tf

time_distributed_layer = tf.keras.layers.TimeDistributed(
    layer=tf.keras.Sequential(
        layers=[tf.keras.layers.Conv2D(filters=16, kernel_size=3)],
    ),
    input_shape=(None, 100, 100, 3),
)
model = tf.keras.Sequential([time_distributed_layer])
model.save('model')  # Same error as before