import math
import tensorflow as tf

input_ = layers.Input(shape=(3))
position = layers.Dense(5, 'sigmoid', name='position')(input_)
out = tf.math.reduce_sum(tf.cast(position> 0.5, tf.float32), axis=-1, keepdims=True, name='final')
model = models.Model(inputs=input_, outputs=[position, out])

model.summary()