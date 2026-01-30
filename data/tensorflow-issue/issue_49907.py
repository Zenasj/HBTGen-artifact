import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
from sklearn.datasets import load_boston
import tensorflow as tf

tf.random.set_seed(0)


def make_model():
    input_layer = tf.keras.layers.Input(shape=(np.shape(x)[1]))
    inner_layer_1 = tf.keras.layers.Dense(
        units=10,
        activation='selu',
        kernel_initializer=tf.keras.initializers.lecun_normal()
    )(input_layer)
    inner_layer_2 = tf.keras.layers.Dense(
        units=10,
        activation='selu',
        kernel_initializer=tf.keras.initializers.lecun_normal()
    )(inner_layer_1)
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation='linear'
    )(inner_layer_2)
    return tf.keras.Model(
        input_layer,
        output_layer
    )

# Get raw data.
raw_features, y = load_boston(return_X_y=True)

# Standardize features.
x = (raw_features-raw_features.mean(axis=0)) / raw_features.std(axis=0)
print(np.std(x, axis=0))

# Reshape targets.
y = np.array(np.reshape(y, newshape=(-1, 1)), dtype=np.float32)

# Training parameters.
optimizer = tf.keras.optimizers.Adam()
optimizer_mp = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
    optimizer, "dynamic"
)
objective = tf.keras.losses.MeanSquaredError()
batch_size = 4
epochs = 5

# Fit via fit method.
train_fit = make_model()
train_fit.summary()
train_fit.compile(optimizer, objective)
train_fit.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

# Fit via gradient tape.
gradient_tape_fit = make_model()
gradient_tape_fit.summary()
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
for epoch in range(0, epochs):
    for step, (x_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = gradient_tape_fit(x_batch, training=True)
            loss = objective(y_batch, predictions)
            scaled_loss = optimizer_mp.get_scaled_loss(loss)
        scaled_grads = tape.gradient(
            scaled_loss, gradient_tape_fit.trainable_weights
        )
        gradients = optimizer_mp.get_unscaled_gradients(
            scaled_grads
        )
        optimizer.apply_gradients(zip(
            gradients, gradient_tape_fit.trainable_weights
        ))
    predictions_via_tape = gradient_tape_fit.predict(x)
    print(
        'Tape MSE: %s' % np.mean(np.power(y-predictions_via_tape, 2))
    )

predictions_via_train = train_fit.predict(x)
print(
    'Fit MSE: %s' % np.mean(np.power(y-predictions_via_train, 2))
)