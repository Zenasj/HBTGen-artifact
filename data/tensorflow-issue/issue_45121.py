# tf.random.uniform((B, 10, 1), dtype=tf.float32) ← Based on Input shape from the issue: Input shape = (10, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate two LocallyConnected1D layers with different implementations to compare behavior
        # implementation=1 works correctly and saves fine
        self.locally_connected_impl1 = tf.keras.layers.LocallyConnected1D(
            filters=1,
            kernel_size=5,
            strides=5,
            implementation=1,
            name="locally_connected_impl1"
        )
        # implementation=3 is known to cause saving issues as described
        self.locally_connected_impl3 = tf.keras.layers.LocallyConnected1D(
            filters=1,
            kernel_size=5,
            strides=5,
            implementation=3,
            name="locally_connected_impl3"
        )

    def call(self, inputs):
        # Forward pass through both implementations
        output_impl1 = self.locally_connected_impl1(inputs)
        output_impl3 = self.locally_connected_impl3(inputs)
        
        # Compute element-wise absolute difference between outputs
        diff = tf.abs(output_impl1 - output_impl3)
        
        # Additionally, compute a boolean mask where they are close within a tolerance
        # This comparison can highlight if the two implementations behave differently
        comparison = tf.math.less_equal(diff, 1e-5)
        
        # Return a dictionary with both outputs, difference, and boolean comparison mask
        # so user can inspect discrepancies
        return {
            "output_impl1": output_impl1,
            "output_impl3": output_impl3,
            "diff": diff,
            "close_match": comparison,
        }


def my_model_function():
    # Return an instance of MyModel, which encapsulates both implementations for comparison
    return MyModel()


def GetInput():
    # Generate input tensor matching the expected input shape of MyModel
    # From the issue example, input shape is (10,1) per batch.
    # Assume batch size 1 for simplicity.
    batch_size = 1
    input_shape = (10, 1)  # sequence length 10, channels 1
    
    # Create a random float32 tensor with values uniformly sampled between 0 and 1
    return tf.random.uniform((batch_size,) + input_shape, dtype=tf.float32)

