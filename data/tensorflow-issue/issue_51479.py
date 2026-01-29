# tf.random.uniform((1, 1), dtype=tf.float32) ← Inferred from example input shape in the issue reproduction code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple sequential model resembling the one from the example in the issue:
        self.dense1 = tf.keras.layers.Dense(units=1, input_shape=(1,))
        self.dense2 = tf.keras.layers.Dense(units=16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=1)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the expected input shape (batch_size=1, features=1)
    # In the issue example, the shape was (1,1)
    return tf.random.uniform((1, 1), dtype=tf.float32)

