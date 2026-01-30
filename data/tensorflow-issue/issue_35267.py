from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

input_a = keras.layers.Input(shape=(10,), name="input_a")
input_b = keras.layers.Input(shape=(20,), name="input_b")
output_a = keras.layers.Dense(1, name="output_a")(input_a)
output_b = keras.layers.Dense(1, name="output_b")(input_b)
model = keras.Model(inputs=[input_a, input_b], outputs=[output_a, output_b])
model.compile(optimizer="sgd", loss={"output_a": None, "output_b": "mse"})

n = 128
input_a = np.ones((n, 10))
input_b = np.ones((n, 20))
output_a = np.ones((n, 1))
output_b = np.ones((n, 1))

dataset = tf.data.Dataset.from_tensor_slices(
    ((input_a, input_b), (output_a, output_b))
).batch(64)

model.fit(dataset)

import tensorflow.keras.backend as K

def null_loss(y_true, y_pred):
    return K.zeros_like(y_true)

loss_fns = [
        loss_fn for loss_fn in model.loss_functions if loss_fn is not None
    ]

import tensorflow as tf

# a model that forks into two independent heads 'one' and 'two'
inputs = tf.keras.layers.Input(shape=(1,))
one = tf.keras.layers.Dense(units=1, name='one')(inputs)
two = tf.keras.layers.Dense(units=1, name='two')(inputs)
model = tf.keras.Model(inputs=inputs, outputs=dict(one=one, two=two))

losses = dict(one=None, two='mse')  # 'one' loss is None, training should not affect weights of 'one'

model.compile(loss=losses, optimizer='adam')

x = tf.data.Dataset.from_tensor_slices([0., 1.])
y = tf.data.Dataset.from_tensor_slices([1., 2.])
yy = tf.data.Dataset.zip(dict(one=y, two=y))  # targets on both heads, but ONLY 'two' has loss
dataset = tf.data.Dataset.zip((x, yy)).batch(1).repeat()

model.fit(dataset, steps_per_epoch=1000, epochs=10)  # should NOT fit 'one' weights, only 'two' weights

# printout shows that, unexpectedly, 'one' weights WERE fitted, 'two' weights WERE NOT fitted
for var in model.trainable_weights: 
    print(f'{var.name}: {var.numpy()}')