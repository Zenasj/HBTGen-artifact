# tf.random.uniform((B, 16), dtype=tf.float32) ← Input shape inferred from example tensor shape (32,16)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple sequential-like structure matching the example:
        # Input (16,) -> Dense(16) -> Dense(16)
        self.dense1 = tf.keras.layers.Dense(16)
        self.dense2 = tf.keras.layers.Dense(16)
    
    def call(self, inputs, training=None):
        # 'training' argument to support typical Keras learning_phase semantics (though unused here)
        x = self.dense1(inputs)
        out = self.dense2(x)
        return out

def my_model_function():
    # Return an instantiated MyModel
    return MyModel()

def GetInput():
    # Produce a random input tensor matching (batch_size, 16)
    # Batch size: 32 as in example
    return tf.random.uniform((32, 16), dtype=tf.float32)

