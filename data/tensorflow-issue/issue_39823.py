# tf.random.uniform((B, 10, 40), dtype=tf.float32) ← inferred input shape from the issue's Conv1D input shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv1D layer with dilation_rate>1 (dilation_rate=2) and 32 filters as per example
        self.conv1d = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            dilation_rate=2,
            padding='same',
            use_bias=False
        )
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.global_max_pool(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel, no pretrained weights given in the issue so default init
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Input shape is (batch_size, 10, 40) - batch size can be any positive integer, we use 1 here
    return tf.random.uniform(shape=(1, 10, 40), dtype=tf.float32)

