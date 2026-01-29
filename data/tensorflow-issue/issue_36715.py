# tf.random.uniform((32, 10), dtype=tf.float64) ← inferred input shape and dtype from the examples

import tensorflow as tf

class Sampler(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # Generate fresh normal noise every call to preserve stochasticity in eager mode
        noise = tf.random.normal(shape=tf.shape(inputs), dtype=inputs.dtype, name="noise")
        return inputs + noise

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate the sampler layer that adds fresh random noise each call
        self.sampler = Sampler()

    def call(self, inputs, training=None):
        # Forward pass adds fresh noise on every call
        return self.sampler(inputs)

def my_model_function():
    # Return a new instance of MyModel
    # This model will generate new noise each time it is called, unlike the flawed fixed-tensor approach
    return MyModel()

def GetInput():
    # Return a random tensor input matching the model's expected shape and dtype
    # Batch size 32, sequence length 10, dtype float64 as per issue example
    return tf.ones(shape=(32, 10), dtype=tf.float64)

