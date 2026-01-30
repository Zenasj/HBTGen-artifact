import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_addons as tfa

potentials = tf.random.uniform((3,4,5))
sequence_lengths = tf.ones((3,))
tag_indices = tf.ones((3,4), dtype=tf.int32)

class CRF(tf.keras.layers.Layer):
    def build(self, input_shape: tf.TensorShape) -> None:
        self.transition_params = self.add_weight(
            shape=(5, 5),
            regularizer=tf.keras.regularizers.l2(0.1),
            name="transitions",
        )
        self.built = True
    
    def call(self, potentials, sequence_lengths):
        x = tfa.text.crf.crf_decode(
            potentials, self.transition_params, sequence_lengths
        )
        y = tfa.text.crf.crf_log_likelihood(
            potentials, tag_indices, sequence_lengths, self.transition_params
        )
        return x, y

crf = CRF()

def foo(potentials, sequence_lengths):
    with tf.GradientTape(persistent=True) as tape:
        x, y = crf(potentials, sequence_lengths)

    return x, y

tf_foo_static =  tf.function(foo)
tf_foo_static(potentials, sequence_lengths)  # works fine

tf_foo_dynamic = tf.function(foo, input_signature=[tf.TensorSpec((None, None, 5)), tf.TensorSpec((None,))])
tf_foo_dynamic(potentials, sequence_lengths)  # the execution is stuck without raising any error