# tf.random.uniform((1, 2), dtype=tf.float32) ← Input shape inferred from issue, 1 sample with 2 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a single Dense layer with 3 output units and sigmoid activation
        self.dense_layer = tf.keras.layers.Dense(
            units=3, 
            activation='sigmoid', 
            input_shape=(2,)
        )

    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense_layer(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor shaped (1, 2) matching the input shape expected by MyModel
    return tf.random.uniform(shape=(1, 2), dtype=tf.float32)

