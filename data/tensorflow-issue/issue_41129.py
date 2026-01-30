import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


batch_size, sequence_length = 3, 5

x_input = tf.keras.layers.Input(
    shape=(sequence_length, 1),
    name='input',
    dtype=tf.float32)

mask_input = tf.keras.layers.Input(
    shape=(sequence_length, ),
    name='mask',
    dtype=tf.bool)


out = tf.keras.layers.LSTM(
    units=256,
    return_sequences=True,
    return_state=False,
)(x_input, mask=mask_input)
hidden_layer_sizes = (256,  256)
for hidden_layer_size in hidden_layer_sizes:
    out = tf.keras.layers.Dense(hidden_layer_size, activation='relu')(out)
out = tf.keras.layers.Dense(1, activation='linear')(out)
model = tf.keras.Model((x_input, mask_input), out)

x = tf.random.uniform(
    (batch_size, sequence_length, x_input.shape[-1]),
    dtype=x_input.dtype)

mask = tf.sequence_mask(
    tf.random.uniform(
        (batch_size, ), minval=0, maxval=sequence_length, dtype=tf.int32),
    maxlen=sequence_length,
)[..., ::-1]


@tf.function(experimental_relax_shapes=True)
def compute_jacobian(x):
    y_true = tf.zeros(batch_size)
    with tf.GradientTape() as tape:
        y = model((x, mask))
        y = tf.reduce_sum(y, axis=1)
        loss = tf.losses.MSE(y_pred=y, y_true=y_true)

    jacobian = tape.jacobian(
        loss,
        model.trainable_variables,
        parallel_iterations=10,
        experimental_use_pfor=False)

    return jacobian

_ = compute_jacobian(x)