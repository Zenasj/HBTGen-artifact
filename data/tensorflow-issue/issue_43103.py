import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

gru = tf.keras.layers.GRU(128)
rng = tf.random.get_global_generator()
inp = rng.normal((8, 64, 256))
msk = tf.concat([tf.ones((4, 64), dtype=tf.bool), tf.zeros((4, 64), dtype=tf.bool)], axis=0)

# works
with tf.device('CPU:0'):
    gru(inp, mask=msk)
# works
with tf.device('GPU:0'):
    gru(inp, mask=tf.ones_like(msk))
# fails
with tf.device('GPU:0'):
    gru(inp, mask=msk)

import tensorflow as tf

class SafeRNN(tf.keras.layers.Wrapper):
    """Wrapper for keras RNN layers avoiding a mask-caused cuda error."""

    def call(self, inputs, mask=None, **kwargs):
        """Run inputs through the wrapped layer."""
        if mask is not None:
            valid = tf.reduce_any(mask, axis=1, keepdims=True)
            mask = tf.where(valid, mask, tf.ones_like(mask))
        return self.layer(inputs, mask=mask, **kwargs)

    def compute_mask(self, inputs, mask=None):
        """Return an output mask tensor."""
        if mask is None:
            return None
        return tf.reduce_any(mask, axis=1)

gru = SafeRNN(tf.keras.layers.GRU(128))
rng = tf.random.get_global_generator()
inp = rng.normal((8, 64, 256))
msk = tf.concat([tf.ones((4, 64), dtype=tf.bool), tf.zeros((4, 64), dtype=tf.bool)], axis=0)

# works
with tf.device('GPU:0'):
    gru(inp, mask=msk)