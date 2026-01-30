from tensorflow import keras
from tensorflow.keras import layers

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU

if __name__ == "__main__":
    with tf.device('/CPU:0'):
        batches = [1] + list(range(1, 10))
        shape = (1000, 512)
        inputs = tf.keras.Input(shape=shape)
        rnn = GRU(shape[-1]//2, return_sequences=True)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=rnn)
        results = []
        for i, batch in enumerate(batches):
            x = tf.ones((batch,) + shape)
            y = model.predict_on_batch(x)
            results.append(y[0])

        for b, x in list(zip(batches, results))[1:]:
            print(b, np.max(np.abs(results[0] - x)))
        if not all(np.allclose(x, results[0]) for x in results[1:]):
            raise ValueError("Varying batch size produces different results")