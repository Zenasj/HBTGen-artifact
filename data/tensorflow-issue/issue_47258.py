# tf.random.uniform((1, 384, 128), dtype=tf.float32) ← inferred input shape from benchmarks and examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the described pattern from the issue:
        # A Dense layer applied to input shape (1, 384, 128)
        # The input is 3D, often with a subsequent reshape removed in TFLite.
        # We model a Dense layer operating on the last dimension 128.
        # 
        # To reflect the discussion about Reshape-FullyConnected-Reshape pattern:
        # Instead of explicit reshape, we rely on Dense handling N-D inputs.
        # 
        # Also, example of two consecutive Dense layers given:
        # Dense(2) with input shape (1, 384, 512),
        # then Dense(3).
        #
        # To fuse the example and provide more generality,
        # implement two Dense layers sequentially,
        # mimicking pattern before and after reshape removal.
        self.dense1 = tf.keras.layers.Dense(
            128, use_bias=True,
            batch_input_shape=(1, 384, 128))
        self.dense2 = tf.keras.layers.Dense(64, use_bias=True)
        # For demonstration: two-layer dense stack supports batch size 1, 3D input.
        
    def call(self, inputs, training=False):
        # Inputs expected shape: (1, 384, 128)
        # Dense layers handle N-D inputs, apply to last dimension.
        x = self.dense1(inputs)
        # According to mobilebert pattern, originally a reshape op might have been here,
        # but it's removed (from the issue), so no explicit reshape is used.
        x = self.dense2(x)
        return x

def my_model_function():
    # Return a model instance
    return MyModel()

def GetInput():
    # Return a random tensor matching input shape expected by MyModel
    # Input shape inferred from linked examples and benchmarks: (1, 384, 128)
    # dtype float32 per TF default for Dense layers
    return tf.random.uniform((1, 384, 128), dtype=tf.float32)

