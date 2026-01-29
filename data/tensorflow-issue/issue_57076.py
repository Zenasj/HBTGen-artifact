# tf.random.uniform((B,), dtype=tf.int32) ← Using a 1D integer tensor as input, compatible with StringLookup

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, small_table_size=10, large_table_size=5_000_000):
        super().__init__()
        # Two StringLookup layers that mimic the original LookupModel behavior
        self.small_lookup = tf.keras.layers.StringLookup(
            vocabulary=[str(i) for i in range(small_table_size)],
            trainable=False,
            mask_token=None,
            oov_token=None,
            name="small_lookup"
        )
        self.large_lookup = tf.keras.layers.StringLookup(
            vocabulary=[str(i) for i in range(large_table_size)],
            trainable=False,
            mask_token=None,
            oov_token=None,
            name="large_lookup"
        )

    def call(self, inputs):
        # The inputs are expected as integer tensor indices; convert them to strings for StringLookup
        # Because StringLookup expects strings, convert int inputs to string tensors
        inputs_str = tf.strings.as_string(inputs)

        # Lookup results from small and large StringLookup layers
        small_result = self.small_lookup(inputs_str)
        large_result = self.large_lookup(inputs_str)

        # Compare the two lookup outputs for equality
        comparison = tf.equal(small_result, large_result)

        # Return a dictionary with individual outputs and their comparison
        return {
            "small_result": small_result,
            "large_result": large_result,
            "are_equal": comparison
        }

def my_model_function():
    # Return an instance of MyModel with default sizes as per the original issue repro
    return MyModel()

def GetInput():
    # Return a random integer tensor simulating input indices in the range [0,9] (small vocab range)
    # Shape: (4,) arbitrary batch of 4 inputs
    # This ensures inputs are valid strings convertible and present in small vocab
    return tf.random.uniform(shape=(4,), minval=0, maxval=10, dtype=tf.int32)

