# tf.random.uniform((BATCH_SIZE, 784), dtype=tf.float32) ← BATCH_SIZE inferred from num_replicas * per-replica batch size

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
        self.pred_layer = layers.Dense(10, name='predictions')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.pred_layer(x)


def my_model_function():
    """
    Return an instance of MyModel with no pretrained weights.
    """
    return MyModel()


def GetInput():
    """
    Return a random input tensor matching the expected input shape for MyModel.
    The original example uses 784-dimensional input vectors (flattened 28x28),
    batched over BATCH_SIZE examples.
    We assume a batch size of 64 as a reasonable default for a single worker.
    """
    BATCH_SIZE = 64  # Assumed typical batch size per worker
    input_shape = (BATCH_SIZE, 784)  # From the original problem input dimension

    # Random uniform float tensor with dtype float32, typical for model input
    return tf.random.uniform(input_shape, dtype=tf.float32)

