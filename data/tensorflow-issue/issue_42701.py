# tf.random.uniform((B, 128), dtype=tf.float32) ← Input shape inferred as batch size B and feature size 128

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Create a non-trainable variable to mimic the "memory bank" on CPU
        # NOTE: In MirroredStrategy scope, placing on CPU by with tf.device('/cpu') 
        # still creates a MirroredVariable replicated per GPU.
        # This is a known limitation as explained in the issue.
        with tf.device('/cpu:0'):
            self.memory = tf.Variable(
                tf.random.uniform([100, 128]), trainable=False, dtype=tf.float32, name="memory_bank"
            )
        
        # Dense layer applied before memory interaction
        self.dense1 = tf.keras.layers.Dense(128)
        
        # Dense layer applied after memory matmul result
        self.dense2 = tf.keras.layers.Dense(128)
    
    def call(self, inputs):
        # Forward pass:
        # 1. Dense layer on input
        x = self.dense1(inputs)
        
        # 2. Matrix multiply with memory bank (transpose memory for matmul)
        # Here we rely on memory being directly used; with MirroredStrategy this becomes replicated.
        res = tf.matmul(x, self.memory, transpose_b=True)
        
        # 3. Dense layer on the result
        output = self.dense2(res)
        
        return output


def my_model_function():
    # Return an instance of MyModel.
    # The model internally creates the memory variable on CPU (but distributed training
    # with MirroredStrategy replicates variables on each GPU, which is a known limitation).
    return MyModel()


def GetInput():
    # Generate a random batch input tensor compatible with MyModel:
    # Shape: (batch_size, 128), dtype: float32
    # Here we pick batch size 8 as a reasonable default for testing.
    batch_size = 8
    return tf.random.uniform((batch_size, 128), dtype=tf.float32)

