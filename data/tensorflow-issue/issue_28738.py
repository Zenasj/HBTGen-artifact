# tf.random.uniform((5, 1), dtype=tf.float32) ← inferred input shape based on fit_generator with input_shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple dense model matching the example: input_shape=(1,), output one unit
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Straightforward forward pass
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel, compiled as in the example
    model = MyModel()
    model.compile(loss="mae", optimizer="adam")
    return model

def GetInput():
    # Return a batch of 5 samples with shape (5,1), matching the example's steps_per_epoch=5
    # Use float32 uniform random values
    return tf.random.uniform((5,1), dtype=tf.float32)

