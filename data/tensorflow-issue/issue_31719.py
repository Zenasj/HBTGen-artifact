from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

tf.keras.backend.set_floatx('float64')


def grad(y, x):
    V = layers.Lambda(lambda z: K.gradients(
        z[0], z[1]), output_shape=[1])([y, x])
    return V


fixed_input = keras.Input(shape=(2,))
a = fixed_input * fixed_input
b = grad(a, fixed_input)

model = keras.Model(inputs=[fixed_input], outputs=[b])

# c = model(tf.constant(2.0, dtype=tf.float64))
c = model.predict(tf.constant([[2.0, 1.0]], dtype=tf.float64))
print(c)

class Toy(tf.keras.Model):
    def call(self, x):
        return x * x

toy = Toy()

x = tf.constant([2.0, 1.0], dtype=tf.float64)
with tf.GradientTape() as tape:
    # Remember to watch non-variable tensor
    tape.watch(x)
    y = toy(x)
print(tape.gradient(y, x))