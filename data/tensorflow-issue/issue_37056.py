# tf.random.uniform((4, 3), dtype=tf.float32) ← inferred input shape from the issue reproduction code
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A Lambda layer that concatenates its input along the last axis.
        # This mimics the original minimal reproducer which used:
        # tf.keras.layers.Lambda(lambda x: tf.concat(tf.nest.flatten(x), axis=-1))
        # Since input is a tensor of shape (4, 3), flattening is trivial here.
        self.lambda_layer = tf.keras.layers.Lambda(lambda x: tf.concat(tf.nest.flatten(x), axis=-1))

    def call(self, inputs):
        return self.lambda_layer(inputs)

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # According to the original example, inputs is of shape (4, 3)
    return tf.random.uniform((4, 3), dtype=tf.float32)

