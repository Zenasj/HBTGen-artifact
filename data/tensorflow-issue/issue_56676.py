# tf.random.uniform((B, 10), dtype=tf.int64) ← Inferred input shape from Demo model input

import tensorflow as tf
import numpy as np

# This function mimics _get_current_replica_id_in_group_sync() from the issue,
# to get the replica id within the MirroredStrategy context.
def _get_current_replica_id_in_group_sync():
    replica_ctx = tf.distribute.get_replica_context()
    if replica_ctx:
        replica_id = replica_ctx.replica_id_in_sync_group
    else:
        # fallback if no replica context
        replica_id = tf.constant(0, dtype=tf.int32)
    return replica_id

# A test function used inside TestLayer call
def test(values):
    global_replica_id = _get_current_replica_id_in_group_sync()
    # Print the replica id tensor, to confirm distributed behavior
    tf.print("global_replica_id: {}".format(global_replica_id))
    # Return a zero tensor with the same shape as input values
    vector = tf.zeros_like(values)
    return vector

class TestLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TestLayer, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        # Call the test function that prints replica id and returns zeros of input shape
        emb_vector = test(values=inputs)
        return emb_vector

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Instantiate the custom TestLayer
        self.test_layer = TestLayer()
        # Dense layer as described, 1 unit, no activation, weights initialized to ones and zeros
        self.dense_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer="ones",
            bias_initializer="zeros"
        )

    def call(self, inputs, training=False):
        # Pass inputs through test_layer which prints replica id and outputs zeros
        vector = self.test_layer(inputs)
        # Then through dense layer for final logits
        logit = self.dense_layer(vector)
        return logit, vector

    def summary(self):
        # Provide a Keras model summary by wrapping call method
        inputs = tf.keras.Input(shape=(10,), dtype=tf.int64)
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

# Helper function to create randomized dataset inputs matching input shape
def GetInput():
    # Following the dataset shape (batch_size, 10) from issue's dataset function
    batch_size = 16  # small batch size for example
    # Input dtype is int64 as per model input and dataset usage
    input_tensor = tf.random.uniform(shape=(batch_size, 10), minval=0, maxval=100, dtype=tf.int64)
    return input_tensor

