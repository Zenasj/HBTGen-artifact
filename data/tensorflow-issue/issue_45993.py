# tf.random.uniform((B, 28*28), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple dense layer to replicate the example's usage
        self.dense = tf.keras.layers.Dense(512, activation='relu', input_shape=(28*28,))

    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input of shape (batch_size=32, input_dim=28*28) matching the input shape of MyModel
    return tf.random.uniform((32, 28*28), dtype=tf.float32)

