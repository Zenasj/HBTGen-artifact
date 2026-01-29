# tf.random.uniform((B, 10), dtype=tf.float32) ← Inferred input shape for model: shape=(None, 10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a small feedforward model similar to the example
        self.dense = tf.keras.layers.Dense(2)
        self.activation = tf.keras.layers.Activation("softmax")
        
    def call(self, inputs):
        x = self.dense(inputs)
        return self.activation(x)

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of random inputs with shape (batch_size, 10)
    # Using batch size 4 arbitrarily; any positive integer batch size would work.
    return tf.random.uniform((4, 10), dtype=tf.float32)

