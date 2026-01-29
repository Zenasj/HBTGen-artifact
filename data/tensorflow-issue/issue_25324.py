# tf.random.uniform((B, 32), dtype=tf.float32) ← input shape inferred from the example data (1000, 32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Model architecture matches the functional API example:
        # Input(shape=(32,))
        # Dense(64, activation='relu') -> Dense(64, activation='relu') -> Dense(10, activation='softmax')
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.out_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out_layer(x)


def my_model_function():
    # Return an instance of MyModel
    # This matches the architecture used in the original issue repro code.
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Based on the example, input shape is (batch_size, 32)
    # Batch size can be arbitrarily chosen; here we pick 16 for demonstration
    return tf.random.uniform((16, 32), dtype=tf.float32)

