# tf.constant(702.89) is scalar input (shape=()), but simplified for demonstration
# Input shape: () scalar float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Single trainable variable as in original code
        self.v3_weight = tf.Variable(123.45, dtype=tf.float32)

    @tf.function(jit_compile=True)
    def call(self, x):
        # Perform multiplication with -0.0 
        # Return result in two modes: naive and with tf.function to demonstrate difference
        # Since direct demonstration is limited here, we emulate the behavior by computing both
        # 1) multiplication without @tf.function (emulated here as direct ops)
        # 2) multiplication with @tf.function(jit_compile=True) - which keeps -0.0 sign
        
        # Multiply the weight by -0.0 explicitly
        multiplied = self.v3_weight * (-0.0)

        # Compute reciprocals to show difference between -0.0 and 0.0 cases:
        # Naive behavior (simulated without jit)
        def eager_multiply():
            return self.v3_weight * (-0.0)

        # With XLA/autocluster jit behavior typically treating -0.0 as +0.0:
        # For simulation, we cast multiplied to +0.0 by abs to emulate autocluster effect
        def xla_behavior():
            return 1.0 / tf.math.abs(multiplied)  # yields +inf instead of -inf

        # Reciprocal with sign preserved (expected with jit_compile=True)
        reciprocal_preserve_sign = 1.0 / multiplied  # yields -inf because multiplied is -0.0

        # reciprocal with sign lost (simulating autocluster behavior)
        reciprocal_autocluster = 1.0 / tf.math.abs(multiplied)  # yields +inf

        # Output dictionary with all values (for demonstration, output as tuple)
        # Note: In practice model returns a single tensor. Here we return a float32 tensor vector with 3 values:
        # [multiplied, reciprocal_preserve_sign, reciprocal_autocluster]
        # This fusion demonstrates differences in sign handling between normal and autoclustered cases.
        out = tf.stack([multiplied, reciprocal_preserve_sign, reciprocal_autocluster])
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Return a scalar tensor matching input shape expected by MyModel.call (scalar float32)
    # From the issue, the input is essentially arbitrary and not used in calculation
    return tf.constant(702.89, dtype=tf.float32)

