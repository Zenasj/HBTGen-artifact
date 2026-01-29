# tf.random.uniform((B, 28, 28), dtype=tf.float32)  ← inferred input shape according to MNIST example in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple MNIST classifier consistent with the example in the issue:
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)
    
    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Return a random input tensor shaped (batch_size, 28, 28) with float32 dtype
    # Batch size is guessed as 32 for this example
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

