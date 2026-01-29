# tf.random.uniform((100, 100), dtype=tf.float32) ← Inferred input shape (batch_size=100, dim_input=100)

import tensorflow as tf

num_iterations = 100  # Matches the issue example, adjustable to test memory usage

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple Dense layer as in original example
        self.custom_mask_layer = CustomMask()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, mask=None):
        # Pass inputs through CustomMask layer (which does nothing in call, only compute_mask)
        x = self.custom_mask_layer(inputs, mask=mask)
        return self.dense(x)

class CustomMask(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomMask, self).__init__()

    def compute_mask(self, inputs, mask=None):
        # Note: The original issue shows a performance and memory problem
        # when calling tf.keras.backend.max repeatedly inside loops.
        # Here, we replicate the logic but avoid actual memory leak by sticking to tf.reduce_max.
        # The original code:
        # batch_size = inputs.shape[0]
        # batch_maxes = tf.keras.backend.max(inputs, axis=1)
        # for batch in range(batch_size):
        #   for i in range(num_iterations):
        #     max = tf.keras.backend.max(batch_maxes[batch])

        # Using tf.reduce_max instead of keras.backend.max (which is alias to reduce_max)
        batch_size = tf.shape(inputs)[0]

        # Compute max per batch element along axis=1 (dim_input axis)
        batch_maxes = tf.reduce_max(inputs, axis=1)  # shape: (batch_size,)

        # To simulate the heavy computation done in the original for loops,
        # we run num_iterations of tf.reduce_max on each scalar in batch_maxes.
        # Doing it in a python loop causes memory issues; instead we use tf.repeat and max:
        # But for faithful reproduction, we must keep structure similar (though it’s inefficient),
        # so we do the loop here in a tf.function-compatible way by repeating max operations.

        # However, to avoid memory issues shown in the original, 
        # we just illustrate the computation without causing leak:
        # We replicate the max operation num_iterations times on each batch_max scalar.
        # Since batch_maxes is shape (batch_size,), max over scalar returns the scalar itself,
        # so the loop is effectively a no-op repeated times.

        # For demonstration, just do a tf.identity multiple times in tf.function
        # We use tf.function and tf.range for loop demonstration if needed.
        
        # We do the repeated max operations within a tf.function safely here:
        @tf.function
        def repeated_max(batch_maxes):
            # Dummy loop to "simulate" the workload without memory leak
            ms = batch_maxes
            for _ in tf.range(num_iterations):
                ms = tf.reduce_max(ms)  # Max over tensor, reduces to scalar
                # To keep shape compatible after each iteration, we turn back to shape (batch_size,)
                ms = tf.fill([batch_size], ms)
            return ms

        # Run the repeated max - this returns a tensor of shape (batch_size,)
        final_maxes = repeated_max(batch_maxes)

        # We do not actually use mask output here, returning None as in original code
        return None

    def call(self, inputs, mask=None):
        return inputs

def my_model_function():
    return MyModel()

def GetInput():
    # Return a tensor shaped (100, 100) matching the input shape of MyModel.
    # Use uniform random floats to simulate realistic data.
    return tf.random.uniform((100, 100), dtype=tf.float32)

