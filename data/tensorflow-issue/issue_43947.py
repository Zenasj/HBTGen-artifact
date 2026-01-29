# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← This example is inferred from the sample keras Dense layer input_shape=[1] (batch varies, feature dim=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model matching the example from the issue:
        # keras.Sequential([Dense(units=1, input_shape=[1])])
        self.dense = tf.keras.layers.Dense(units=1, input_shape=(1,))
        
    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Return a random input tensor with the shape expected by MyModel
    # The Keras input_shape=[1] means input is (batch_size, 1)
    # Infer batch size = 8 for example
    batch_size = 8
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)

