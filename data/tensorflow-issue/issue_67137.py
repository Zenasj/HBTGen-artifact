# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from training code and model description

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the model structure as described:
        # Input: (None, 32, 32, 3)
        # Conv2D with filters=5, kernel_size=13x13, tanh activation, bias, random_uniform init
        self.conv = tf.keras.layers.Conv2D(
            filters=5,
            kernel_size=(13, 13),
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
            dilation_rate=(1, 1),
            activation="tanh",
            use_bias=True,
            kernel_initializer="random_uniform",
            bias_initializer="random_uniform"
        )
        
        # ReLU activation capped at max_value=0.08354582293069757
        self.capped_relu = tf.keras.layers.ReLU(max_value=0.08354582293069757)
        
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        
        # Dense layer with 10 units, linear activation, no bias, random_uniform kernel init
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation="linear",
            use_bias=False,
            kernel_initializer="random_uniform",
            bias_initializer="random_uniform"  # Although bias not used, keep for symmetry
        )
        
        # Reshape to (10,)
        self.reshape = tf.keras.layers.Reshape((10,))
        
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.capped_relu(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.reshape(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # The original issue referenced loading weights from a file, but file is unavailable.
    # We will initialize weights randomly as seed is set outside if needed.
    # User can load weights externally if needed.
    return model

def GetInput():
    # Returns a random tensor input that matches the expected input for MyModel
    # Based on training code: shape = (batch_size, 32, 32, 3)
    # Batch size can be arbitrary; use 1 for testing call
    return tf.random.uniform(shape=(1, 32, 32, 3), dtype=tf.float32)

