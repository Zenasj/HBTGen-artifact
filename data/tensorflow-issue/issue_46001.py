# tf.random.uniform((B, 20), dtype=tf.float16) ← Input shape inferred from example input X.shape=(1,20) in the issue reproduction code.

import tensorflow as tf
import tensorflow_addons as tfa

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # WeightNormalization wrapper around a Dense layer with 10 units, ReLU activation.
        # This mirrors the code snippet in the issue.
        # Note: Under mixed precision, internal computations should be done in float32 to avoid dtype mismatch.
        # As a mitigation, we will cast inputs inside call to float32 for moments computation internally,
        # which is consistent with suggested fixes in the issue discussion.
        self.wn_dense = tfa.layers.WeightNormalization(
            tf.keras.layers.Dense(10, activation="relu")
        )
    
    def call(self, inputs, training=False):
        # Because the underlying issue is about mixed precision,
        # and the WeightNormalization layer expects float32 math internally,
        # we ensure input to WeightNormalization is cast to float32.
        # This is a practical workaround aligning with the suggestion that
        # math inside the layer should run in float32 even if inputs are float16.
        inputs_fp32 = tf.cast(inputs, tf.float32)
        # Apply WeightNormalization Dense layer
        out = self.wn_dense(inputs_fp32, training=training)
        # Cast output back to inputs dtype (mixed_precision policy dtype, often float16)
        out = tf.cast(out, inputs.dtype)
        return out

def my_model_function():
    # Creates and returns an instance of MyModel.
    return MyModel()

def GetInput():
    # Return a random input tensor compatible with MyModel's expected input.
    # The example input in the issue was shape (1, 20) with dtype float16 under mixed precision.
    # To match the inference in the initial comment, we provide a batch of 1, feature size 20.
    # We use float16 as the default dtype to simulate mixed precision environment.
    return tf.random.uniform((1, 20), dtype=tf.float16)

