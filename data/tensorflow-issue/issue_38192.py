# tf.random.uniform((B,), dtype=tf.float32) ← The input is a 1D tensor since the example uses tf.data.Dataset.from_tensor_slices([0.,1.,2.,3.]) which yields scalar floats

import tensorflow as tf

# The problem discussed regards compatibility of Dataset with attr-based classes vs namedtuples.
# To represent that in a model, we can create two submodels:
# - One that processes inputs wrapped as a namedtuple-like structure.
# - One that processes inputs wrapped as an attr-based dataclass-like structure.
#
# Since TensorFlow Dataset supports namedtuple mapping but not attr class mapping,
# our model will implement both and do a comparison between their outputs.
#
# In this simplified example, both submodules just perform identity maps
# but via different wrappers, mimicking how tf.function can handle attr classes and namedtuples.
#
# Output: a boolean tensor indicating whether their processed outputs match elementwise.

from collections import namedtuple

# Mimic the attr "frozen" class with tf.Module for this example
class DataContainer(tf.Module):
    def __init__(self, data):
        super().__init__()
        self.data = data

# Named tuple container
NamedDataContainer = namedtuple('NamedDataContainer', ['data'])

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights needed for this demo
        # Define identity-like layers to simulate passing data through submodels
        self.namedtuple_layer = tf.keras.layers.Lambda(lambda x: x.data)
        self.attr_layer = tf.keras.layers.Lambda(lambda x: x.data)
    
    def call(self, x):
        """
        x is expected to be a tf.float32 tensor of shape (B,) representing a batch of scalars
        
        We simulate two pipelines:
        1) Wrap input as NamedDataContainer => process with namedtuple_layer
        2) Wrap input as DataContainer (attr-like) => process with attr_layer
        
        Then return a boolean tensor indicating if the outputs are equal.
        """
        # Wrap inputs
        named_input = NamedDataContainer(data=x)
        attr_input = DataContainer(data=x)
        
        # Extract data via subtmodules
        named_out = self.namedtuple_layer(named_input)  # should just get back x
        attr_out = self.attr_layer(attr_input)          # should just get back x
        
        # Compare outputs elementwise with a tolerance suitable for floats
        comparison = tf.math.abs(named_out - attr_out) < 1e-6
        return comparison

def my_model_function():
    return MyModel()

def GetInput():
    # Return a batched 1D tensor matching the expected input shape and type: (B,)
    # Use batch size 4 to mirror example dataset slices: [0., 1., 2., 3.]
    input_tensor = tf.random.uniform(shape=(4,), minval=0.0, maxval=10.0, dtype=tf.float32)
    return input_tensor

