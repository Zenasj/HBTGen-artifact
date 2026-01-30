import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

# see: https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

def create(seed_a, seed_b):
    inp = tf.keras.Input(shape=(5,))
    x = inp / inp # == 1.0
    a = tf.random.uniform(tf.shape(x), dtype=x.dtype, minval=-5, maxval=5, seed=seed_a) * x
    b = tf.random.uniform(tf.shape(x), dtype=x.dtype, minval=-5, maxval=5, seed=seed_b) * x
    return tf.keras.Model(inputs=[inp], outputs=[a, b])

def compare(seed_a, seed_b):
    print(f"## RUNNING {seed_a} {seed_b} ##")
    data = np.ones((1, 5), np.float32)

    model = create(seed_a, seed_b)

    eager = model(data)
    graph = model.predict(data)

    for i, (a, b) in enumerate(zip(eager, graph)):
        print(f'#{i}, A: {a.numpy()}, B: {b}')

tf.random.set_seed(314159)
compare(123, 124)

tf.random.set_seed(314159)
compare(123, 123)