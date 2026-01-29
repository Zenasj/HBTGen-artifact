# tf.random.uniform((B,), dtype=tf.float32) ← Based on original input spec of shape [None] and tf.float32 for argument "a"

import tensorflow as tf
from typing import List, Dict, Union, ByteString

class MyModel(tf.keras.Model):
    """
    This fused model illustrates usage of a tf.lookup.StaticHashTable that must be tracked properly
    as an attribute initialized outside any tf.function decorated method to allow SavedModel export
    without the "untracked resource" error.

    We reconstruct a pattern from the issue:
    - The StaticHashTable resource is created once during __init__ and assigned to self.table,
      thus making it tracked by TensorFlow's object tracking.
    - The query method is tf.function decorated.
    - Input tensor is a float32 1D vector as seen in original example, here simply passed through.
    
    This example demonstrates the minimal working code to avoid the SavedModel export error.
    """

    def __init__(self):
        super(MyModel, self).__init__()

        # Define keys and values as constants; StaticHashTable is assigned here so TF tracks it.
        keys_tensor = tf.constant([1, 2], dtype=tf.int64)
        vals_tensor = tf.constant([3, 4], dtype=tf.int64)

        # Initialize StaticHashTable to perform lookups
        initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
        self.table = tf.lookup.StaticHashTable(initializer, default_value=-1)

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32, name="a")])
    def call(self, a):
        """
        The forward call simply returns `a` tensor unchanged, but has inside access to self.table.
        Typically, you would lookup inside the table or include it into your computations.
        Here, to keep to original example's minimalism, just demonstrating usage without errors.

        If you want to demonstrate a lookup, you can cast `a` to int64 (or provide int64 input)
        and do a lookup, but original snippet did not do so in output.
        """
        # Example: If input needs to be looked up in the table (cast float to int64 indices)
        # Here commented out since original returned `a` unchanged
        # int_keys = tf.cast(a, tf.int64)
        # looked_up_values = self.table.lookup(int_keys)
        # return looked_up_values

        return a


def my_model_function():
    """
    Construct and return an instance of MyModel.
    The StaticHashTable resource is tracked as part of the model instance.
    """
    return MyModel()


def GetInput():
    """
    Return a random input compatible with MyModel call method input signature:
    - Shape: [None], meaning a 1D vector of float32 with dynamic length.
    For demonstration, create a batch of size 4 random floats.
    """
    batch_size = 4  
    # 1D tensor of floats
    return tf.random.uniform((batch_size,), minval=0, maxval=10, dtype=tf.float32)

