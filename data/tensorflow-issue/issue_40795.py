# tf.random.uniform((1, 1), dtype=tf.float32) ← inferred input from tf.TensorSpec(shape=[1,1])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense_layer = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=None):
        # The original example has an `if training:` clause, but both branches do the same.
        # We keep the if clause to reflect the original model structure.
        # The training argument can be True/False or a tf.bool tensor.
        # Note: The original issue was about gradient failures with if condition in saved models.
        if training:
            return self.dense_layer(inputs)
        else:
            return self.dense_layer(inputs)

def my_model_function():
    # Return an instance of MyModel.
    # The original example compiled the call method with tf.function and a concrete function.
    # For compatibility with XLA and saving/loading, we rely on keras Model and tf.function outside here.
    return MyModel()

def GetInput():
    # Return a random tensor matching the model input shape: [1, 1], dtype float32.
    # This matches the original example's input spec.
    return tf.random.uniform(shape=(1, 1), dtype=tf.float32)

