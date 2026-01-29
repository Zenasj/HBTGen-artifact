# tf.random.uniform((B, 3), dtype=tf.float32) ← The input shape is (batch_size, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple feed-forward network as described, with layers:
        # Dense(5, activation='relu') -> Dense(5, activation='relu') -> Dense(1, activation='tanh')
        # Note: Original code uses Z.relu and Z.tanh (likely a typo or shorthand for tf.keras.activations)
        self.dense1 = tf.keras.layers.Dense(5, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(5, activation='relu', name='dense2')
        self.dense3 = tf.keras.layers.Dense(1, activation='tanh', name='dense3')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected shape (batch size 1 for example)
    # Using uniform distribution [0,1)
    return tf.random.uniform((1, 3), dtype=tf.float32)

