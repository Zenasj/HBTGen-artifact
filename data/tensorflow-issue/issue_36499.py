# tf.random.uniform((B, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Lambda layer mimicking the intended tf.fill behavior but correctly using input shape and a workaround
        # Note: tf.fill with dynamic shape from symbolic tensors inside Lambda causes errors,
        # so we use tf.ones_like multiplied by the fill value instead as a workaround.
        self.fill_layer = tf.keras.layers.Lambda(
            lambda x: tf.ones_like(x) * 2.5,
            name="fill_with_2.5"
        )

    def call(self, inputs):
        # Return a tensor of same shape as inputs, filled with 2.5 values
        return self.fill_layer(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input with shape (batch_size=10, 1) matching the expected input shape
    # dtype=tf.float32 is the default dtype used in TF
    batch_size = 10
    input_shape = (batch_size, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

