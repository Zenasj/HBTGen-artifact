# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape is (batch_size, 1) as from the repro example with np.ones([i, 1])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model replicated from the original example: a single Dense(1) layer with input shape [1]
        self.dense = tf.keras.layers.Dense(1, input_shape=(1,))
        
        # To demonstrate the "fused" comparison logic inspired by the issue context:
        # We'll create two functionally identical modules and compare their outputs when called on the same input.
        self.modelA = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        self.modelB = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        # Initialize modelB weights to be identical to modelA to ensure predictable comparison
        # This requires a build or dummy call first to create variables
        dummy_input = tf.zeros((1,1))
        self.modelA(dummy_input)
        self.modelB(dummy_input)
        # Copy weights from modelA to modelB
        for wa, wb in zip(self.modelA.weights, self.modelB.weights):
            wb.assign(wa)
        
    @tf.function
    def call(self, inputs):
        # Run both submodels on the input
        outA = self.modelA(inputs)
        outB = self.modelB(inputs)
        # Compare outputs elementwise - difference absolute value
        diff = tf.abs(outA - outB)
        # For demonstration:
        # Return a boolean tensor indicating if outputs are close within a tolerance (e.g., 1e-6)
        close = tf.less_equal(diff, 1e-6)
        # Return a dictionary with A output, B output, difference, and close mask
        # This simulates the fusion and comparison logic from the issue context
        return {
            "modelA_output": outA,
            "modelB_output": outB,
            "difference": diff,
            "close": close
        }

def my_model_function():
    # Return an instance of MyModel
    # Weight initialization and syncing is done inside __init__
    return MyModel()

def GetInput():
    # Return a random tensor input matching (batch_size, 1)
    # batch_size here is variable, we choose a random int between 1 and 10 to simulate varying batch sizes
    batch_size = tf.random.uniform([], minval=1, maxval=11, dtype=tf.int32)
    # Use tf.random.uniform to generate float32 input tensor
    input_tensor = tf.random.uniform((batch_size, 1), dtype=tf.float32)
    return input_tensor

