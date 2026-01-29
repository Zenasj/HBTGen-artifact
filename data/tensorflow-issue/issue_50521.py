# tf.random.uniform((3,), dtype=tf.float32) ← Input shape inferred from SIZE=3 and tf.TensorSpec(shape=(SIZE,), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def build(self, input_shape):
        # Weight vector of shape (3,)
        self.w = self.add_weight(shape=(3,), trainable=True, initializer="zeros")

    @tf.function(jit_compile=True)
    def call(self, input):
        # Increment internal weight by input tensor (in-place mutation via assign_add)
        # This matches original behavior: self.w.assign_add(input)
        self.w.assign_add(input)
        # Return input unmodified as in original call method
        return input

def my_model_function():
    # Return an instance of MyModel, initialized with zeros
    return MyModel()

def GetInput():
    # Return a random float32 tensor of shape (3,) matching expected input
    # Use tf.random.uniform to generate values in [0,1)
    return tf.random.uniform((3,), dtype=tf.float32)

