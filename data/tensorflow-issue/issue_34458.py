# tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single dense layer with relu activation, input shape (1,)
        self.dense = tf.keras.layers.Dense(32, activation='relu')
    
    def call(self, inputs):
        # inputs: a single float tensor of shape (batch_size, 1)
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel with initialized weights
    return MyModel()

def GetInput():
    # The original dataset consisted of scalar floats; input shape is (batch_size, 1)
    # Generate batch of 4 random positive floats avoiding zero to prevent divide-by-zero errors seen in the issue.
    # Using uniform float samples [0.1, 2.0)
    batch_size = 4
    return tf.random.uniform((batch_size, 1), minval=0.1, maxval=2.0, dtype=tf.float32)

