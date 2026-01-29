# tf.random.uniform((6, 1), dtype=tf.float32) ← input shape inferred from the original numpy input X (6 samples, 1 feature each)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the simple Sequential flow from the issue:
        # 1) a Lambda layer that logs inputs (using tf.print)
        # 2) a Dense layer with 1 unit as output
        self.log_inputs = tf.keras.layers.Lambda(self._log_inputs)
        self.dense = tf.keras.layers.Dense(1)

    def _log_inputs(self, inputs):
        # This replicates the logging function shown in the issue,
        # printing inputs before passing them forward transparently.
        tf.print(inputs)
        return inputs

    def call(self, inputs, training=False):
        x = self.log_inputs(inputs)
        x = self.dense(x)
        return x

def my_model_function():
    # Returns an instance of the above model, untrained weights initialized by default
    return MyModel()

def GetInput():
    # Inputs similar to original numpy array: 6 samples, 1 feature each
    # Values range 0 to 5, float32 dtype.
    # For generality, we produce a batch dimension (which is standard for Model).
    # The input tensor shape is (batch_size, 1).
    # We pick batch_size as 2 to match the batch size used in the original code.
    X = tf.constant([[0.0],[1.0],[2.0],[3.0],[4.0],[5.0]], dtype=tf.float32)
    # For demonstration, return first 2 samples as batch size 2 tensor.
    # This supports feeding directly into the model.
    return X[:2]

