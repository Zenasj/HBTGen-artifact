import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import psutil
import time
import os

def mem_usage_str():
    process = psutil.Process(os.getpid())
    gb =  process.memory_info().rss / (1024.**3)
    return ' [mem_usage = {:5.2f} GB]'.format(gb)

if int(tf.__version__.split('.')[0]) < 2:
    """
    Patch to fix TF/numpy1.20 compatibility issue
    """
    from   tensorflow.math          import reduce_prod
    from   tensorflow.python.ops    import array_ops

    def _constant_if_small(value, shape, dtype, name):
        try:
            if reduce_prod(shape) < 1000:  # monkey patch
                return array_ops.constant(value, shape=shape, dtype=dtype,
                                          name=name)
        except TypeError:
            # Happens when shape is a Tensor, list with Tensor elements, etc.
            pass
        return None

    array_ops._constant_if_small = _constant_if_small
    """
    End of patch
    """

def build_model():
    inputs = [tf.keras.layers.Input(shape=(300, 6), name='input_layer')]
    current_layer = inputs[0]

    current_layer = tf.keras.layers.LSTM(
        50,
        dropout=0.1,
        recurrent_dropout=0.1,
        return_sequences=False,
        name='lstm',
    )(current_layer)

    current_layer = tf.keras.layers.Dense(1)(current_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model = tf.keras.models.Model(inputs=inputs, outputs=current_layer)
    model.compile(loss='mse', optimizer=optimizer)

    return model

def run(model, X, y, n_epochs):
    tot_time = 0.

    print(f'Memory usage before training' + mem_usage_str())
    for i in range(n_epochs):
        start = time.time()
        model.fit(X, y, epochs=1, batch_size=4096, verbose=0)
        tot_time += time.time() - start
        print(f'Memory usage after epoch {i}' + mem_usage_str())

    print(f'Avg. time = {tot_time / n_epochs} seconds')

def run_example(p, n_epochs):
    import numpy as np

    model = build_model()
    X = np.random.randn(2 ** p, 300, 6)
    y = np.random.randn(2 ** p)

    run(model, X, y, n_epochs)

def main():
    run_example(
        16, # 2 ** 16 samples
        20, # 10 epochs
    )


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()