from metrics import r2_score


def create_model(input_shape, output_shape):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(16, activation='relu'),
        layers.Dense(output_shape)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=[r2_score])

    return model

from sklearn.metrics import r2_score as skl_r2_score


def r2_score(y_true, y_pred):
    return skl_r2_score(y_true.numpy(), y_pred.numpy())

from sklearn.metrics import r2_score as skl_r2_score


def r2_score(y_true, y_pred):
    return skl_r2_score(y_true.numpy(), y_pred.numpy())

import tensorflow as tf


def r2(y_true, y_pred):
    total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    r_squared = tf.subtract(tf.cast(1, tf.float32), tf.divide(unexplained_error, total_error))

    return r_squared