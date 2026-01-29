# tf.random.uniform((1024, 10), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple fully connected model matching the issue input/output shape (input 10 features)
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor input matching the expected input shape [batch_size, 10]
    # Using batch size 1024 to be consistent with the original example
    batch_size = 1024
    input_shape = (batch_size, 10)
    return tf.random.uniform(input_shape, dtype=tf.float32)

