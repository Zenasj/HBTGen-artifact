# tf.random.uniform((B, 4), dtype=tf.float32) ← Input shape inferred as [batch_size, 4] from the example with the Keras Dense layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple Dense layer with output dimension 5, as in the example
        self.final_projection = tf.keras.layers.Dense(5, name="final_projection")

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
    def call(self, sample_input):
        # Apply the Dense layer
        x = self.final_projection(sample_input)
        # Compute mean and std over all elements of x
        mean = tf.math.reduce_mean(x)
        std = tf.math.reduce_std(x)
        return {"mean": mean, "std": std}

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor of shape [batch_size, 4], batch_size=3 chosen arbitrarily
    return tf.random.uniform(shape=(3, 4), dtype=tf.float32)

