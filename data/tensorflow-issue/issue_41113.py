# tf.random.uniform((batch_size, ), dtype=tf.float32) ← Input shape inferred from the minimal example (shape [10]) in get_dataset()

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using SyncBatchNormalization as in the issue repro
        self.sync_bn = tf.keras.layers.experimental.SyncBatchNormalization()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.sync_bn(inputs, training=training)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # The original get_dataset creates inputs of shape [10] with batch size 1
    # but SyncBatchNormalization expects at least rank 2 inputs.
    # The original example used x = tf.zeros([10], ...) then a batch(1)
    # so input tensor shape is (batch, 10)
    #
    # So generate a random tensor with shape (1, 10) and dtype float32 to match expected input.
    # This matches the original shape where each sample is vector of length 10, batch size 1.

    batch_size = 1
    feature_dim = 10
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

