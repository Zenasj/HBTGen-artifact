# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Create layers only once, during initialization to avoid variable creation on call.
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input compatible with MyModel input shape (batch, 32, 32, 3)
    # Using dtype float32 as typical for image input tensors.
    batch_size = 1  # typical single batch; can be adjusted as needed
    input_shape = (batch_size, 32, 32, 3)
    return tf.random.uniform(input_shape, dtype=tf.float32)

