from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

tf.config.set_soft_device_placement(True)

RANGE=80
SIZE=int(1024*3)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model = tf.keras.Sequential([tf.keras.layers.Dense(SIZE) for _ in range(RANGE)])


@tf.function()
def func():
    with tf.device("/gpu:0"):
        for _ in range(1):
            print("STEP", _)
            with tf.GradientTape() as t:
                inp = tf.ones([2, SIZE], tf.float32)
                y = model(inp)
                gradients = t.gradient(y, model.trainable_weights)
                opt.apply_gradients(zip(gradients, model.trainable_weights))
        return gradients

print(func())