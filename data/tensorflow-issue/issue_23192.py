# tf.random.uniform((B, H, W, C), dtype=...) ← Input shape and dtype are unknown from the issue; 
# the model deals with checkpoint restore of variables with shape [2,2], so we assume input shape (2,2) tensor for illustration

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create variables resembling the original variables from the checkpoint example
        # Using tf.Variable for TF2 style instead of deprecated tf.get_variable
        self.variable = tf.Variable(tf.random.uniform([2, 2]), name="variable")
        
        # dict with only ints (non-checkpointable)
        self.dict = {
            "test": 1,
            "test2": 2,
        }
        
        # dict with mixed int and variable
        self.dict_var = {
            "test": 1,
            "test2": tf.Variable(tf.random.uniform([2, 2]), name="dict_var")
        }
        
        # nested dict with variable
        self.dict_nested_var = {
            "test": 1,
            "test2": {
                "var": tf.Variable(tf.random.uniform([2, 2]), name="dict_nested_var")
            }
        }
        
        # list with variable
        self.list_var = [tf.Variable(tf.random.uniform([2, 2]), name="list_var")]
        
        # new dict with only primitive data, simulating new checkpointable data structures that do not contain variables
        self.dict_new = {
            "test": 1,
            "test2": 2
        }
        
    @tf.function(jit_compile=True)
    def call(self, inputs):
        # For demonstration, we just output the sum of all variables to have a tensor output
        # since the original issue revolves around checkpoint handling, not forward pass
        total = tf.zeros(())
        total += tf.reduce_sum(self.variable)
        total += tf.reduce_sum(self.dict_var["test2"])
        total += tf.reduce_sum(self.dict_nested_var["test2"]["var"])
        total += tf.reduce_sum(self.list_var[0])
        # The dict and dict_new have no variables and so do not contribute to computation
        
        # Return total broadcasted to input shape's batch dimension for compatibility
        batch_dim = tf.shape(inputs)[0]
        return tf.fill([batch_dim, 1], total)  # shape: (batch_size, 1)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a batch of random inputs matching shape (batch, 2, 2) to be compatible 
    # with assumed input used in call (batch dimension required for the output shape)
    # Using float32 as default dtype
    batch_size = 4  # chosen arbitrarily
    return tf.random.uniform((batch_size, 2, 2), dtype=tf.float32)

