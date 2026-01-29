# tf.random.uniform((B, 20), dtype=tf.float32) ← Input shape is (batch_size, 20) as per the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example: Input shape (20,)
        # Layers: BatchNormalization followed by Dense(2)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        # Forward pass matching the example build_and_compile_model()
        x = self.batch_norm(inputs, training=training)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Return a random tensor with shape (B, 20)
    # Using batch size = 8 as an example (arbitrary)
    batch_size = 8
    return tf.random.uniform((batch_size, 20), dtype=tf.float32)

