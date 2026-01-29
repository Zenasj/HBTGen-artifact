# tf.random.uniform((B, 1, 1024), dtype=tf.float32)  # inferred input shape and type from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the model described in the issue: 
        # Input shape: (1, 1024) → Dense(1024) → Dense(1)
        self.dense1 = tf.keras.layers.Dense(1024)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default initialization.
    # (Weights are randomly initialized by default.)
    return MyModel()

def GetInput():
    # Return a random tensor input compatible with MyModel call.
    # According to the issue: input shape is (1, 1024) per batch item,
    # and batch dimension appears used in the example as 1.
    # We add batch dim as first dimension.
    # dtype float32 as seen in the example code.
    return tf.random.uniform(shape=(1, 1, 1024), dtype=tf.float32)

# Additional notes based on the issue:
# - Input is 3D tensor (batch_size, 1, 1024) per the example input of shape (1, 1024)
#   wrapped in a batch dimension.
# - Model is two Dense layers sequentially applied.
# - The primary issue was memory leakage on repeated save/load, but here we
#   provide minimal model and input as per requirements.
# - The model and input are compatible with TensorFlow 2.20.0 and can be used
#   with tf.function(jit_compile=True).

# Example usage (do not include test code as per instructions):
#
# model = my_model_function()
# x = GetInput()
# y = model(x)
# print(y)
#
# @tf.function(jit_compile=True)
# def compiled(x):
#     return model(x)

