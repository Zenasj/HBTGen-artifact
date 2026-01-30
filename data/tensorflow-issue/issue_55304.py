import random
from tensorflow import keras
from tensorflow.keras import layers

py
import tensorflow as tf
import tensorflow.keras.layers as layers

model = tf.keras.Sequential([
    layers.Conv2D(8, kernel_size=(5, 5), padding='same', input_shape=(24, 24, 3)),
    layers.MaxPooling2D(pool_size=(24, 24), padding='valid'),
    layers.Flatten(),
    layers.Dense(3),
])

@tf.function
def run_episode(
        model: tf.keras.Model,
        max_steps: int):
    
    outs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for t in tf.range(max_steps):
        ins = tf.random.uniform((16, 24, 24, 3))
        out = tf.recompute_grad(model)(ins)
        outs = outs.write(t, out)
            
    return outs.stack()

with tf.GradientTape() as tape:
    lp = run_episode(model, max_steps=500)
    sigma = tf.reduce_sum(lp)
g = tape.gradient(sigma, model.trainable_variables)