# tf.random.uniform((1, 32, 32, 16), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We replicate the behavior as per the issue:
        # A Dense layer with units=0 is allowed in TF 1.15 which is unreasonable.
        # This model demonstrates that situation.
        # We'll create a Dense layer with units=0.
        # Note: Normally Dense(units=0) is invalid and should raise an error,
        # but per TF 1.15 this was allowed and no special handling was done.
        self.dense_zero_units = tf.keras.layers.Dense(units=0)

    def call(self, inputs):
        # Forward pass through the Dense layer with units=0
        # TensorFlow 1.15 silently accepts this and outputs a tensor with shape
        # [batch_size, 0], which is an empty dimension.
        return self.dense_zero_units(inputs)

def my_model_function():
    # Return an instance of MyModel, with no special initialization needed
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching the input shape used in the issue:
    # shape: (1, 32, 32, 16)
    # dtype: float32 as default in TensorFlow Keras inputs
    return tf.random.uniform((1, 32, 32, 16), dtype=tf.float32)

