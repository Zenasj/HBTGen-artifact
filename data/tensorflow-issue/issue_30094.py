# tf.random.uniform((B, 784), dtype=tf.float32) ← input shape inferred from MNIST flattened (784,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture based on the issue's example
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.pred_layer = tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.pred_layer(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # MNIST flattened input shape (batch_size, 784), here batch size 64 as used in the example
    # Generate random floats simulating normalized pixel intensities in [0,1]
    batch_size = 64
    input_shape = (batch_size, 784)
    return tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)

