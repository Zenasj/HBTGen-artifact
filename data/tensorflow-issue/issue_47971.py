# tf.random.uniform((B, 100), dtype=tf.float64) ← input batches shaped (batch_size, n_features) with float64 dtype

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # According to the original issue, input shape is (None, 100), output targets 5.
        # Using float64 throughout to match the original numpy data type.
        self.dense_1 = tf.keras.layers.Dense(
            100, activation='relu', name='dense_1', dtype=tf.float64)
        self.dense_2 = tf.keras.layers.Dense(
            100, activation='relu', name='dense_2', dtype=tf.float64)
        self.dense_3 = tf.keras.layers.Dense(
            100, activation='relu', name='dense_3', dtype=tf.float64)
        self.dense_4 = tf.keras.layers.Dense(
            100, activation='relu', name='dense_4', dtype=tf.float64)
        self.output_layer = tf.keras.layers.Dense(
            5, activation='linear', name='output', dtype=tf.float64)

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.output_layer(x)
        return x

def my_model_function():
    # Returns a freshly instantiated model with default initialization.
    return MyModel()

def GetInput():
    # Return a random input tensor matching the input shape (batch_size=100, n_features=100) with float64 dtype
    # Batch size chosen to be 100 (same as original batch_size)
    batch_size = 100
    n_features = 100
    return tf.random.uniform(
        shape=(batch_size, n_features), dtype=tf.float64)

