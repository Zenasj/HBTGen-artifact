# tf.random.uniform((None,)) ← Input shape and dtype are inferred as simple 1D tensor of unknown length with numeric values

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model encapsulates the example related to tf.Module variable dictionary iteration issue.
        # Since the original issue involved a custom dict subclass breaking tf.Module.variables access,
        # here we instead demonstrate a safe tf.Module subclass that uses standard dict keys iteration internally.
        #
        # To align with the issue context, we define a submodule with a tf.Variable and a dictionary attribute,
        # and safely expose variables without triggering KeyError.
        
        self.my_var = tf.Variable([1, 2, 3], dtype=tf.int32)  # example variable
        
        # Instead of the problematic MyNamedTuple,
        # use a standard dict attribute to simulate attributes stored in a dictionary:
        self.attrs_dict = {'arg': 'hello'}
    
    def call(self, x):
        # Since the original example was about variable tracking,
        # this call method will just return the input multiplied by self.my_var's first element to make it functional.
        return x * tf.cast(self.my_var[0], x.dtype)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random 1D float tensor input compatible with MyModel call
    # Since MyModel multiplies input by an int32 scalar variable value,
    # input dtype should be float32 or compatible.
    #
    # Shape is (5,) arbitrarily chosen for demonstration.
    return tf.random.uniform((5,), dtype=tf.float32)

