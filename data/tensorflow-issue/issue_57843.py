# tf.random.uniform((5,), dtype=tf.float32) ← Input shape inferred from the examples in the issue (1D tensor with 5 elements)

import tensorflow as tf

class ModuleLevel2(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Using a TensorArray to simulate dynamic state storage,
        # but since TensorArray can't be a member variable for tf.function graph mode,
        # we simulate this with a tf.Variable holding a 3D tensor:
        # shape: (max_steps, batch=1, features=5)
        # We'll accumulate inputs dynamically by concatenation to the variable.
        # This is a workaround to have "state" as a variable that can grow.
        self.state = tf.Variable(
            initial_value=tf.zeros((0, 5), dtype=tf.float32),
            trainable=False,
            shape=tf.TensorShape([None, 5]),
            use_resource=True,
            dtype=tf.float32,
            aggregation=tf.VariableAggregation.NONE
        )
    
    def call(self, x):
        # x shape: (5,) tensor (vector)
        # Append new input x to state variable along axis 0
        # Since tf.Variable does not support dynamic resize in graph mode,
        # we create a new tensor stacking old state plus new x,
        # and assign back to the variable.
        # NOTE: This is not very efficient, but simulates dynamic state.

        # Expand dims to (1, 5) to concatenate with state (N,5)
        x_exp = tf.expand_dims(x, axis=0)
        new_state = tf.concat([self.state, x_exp], axis=0)
        # Assign new_state to self.state variable
        # Using assign to update variable dynamically
        self.state.assign(new_state)
        # Create output by stacking all states (already in variable)
        # Identity op as example in original code
        out = tf.identity(self.state)
        tf.print("ModuleLevel2 state:", out)
        return out

class ModuleLevel1(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.module2 = ModuleLevel2()

    def call(self, x):
        return self.module2(x)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.module1 = ModuleLevel1()

    def call(self, x):
        return self.module1(x)

def my_model_function():
    # Return the model instance.
    # Note: the state variable inside will accumulate inputs on each call.
    return MyModel()

def GetInput():
    # Returns a random input tensor shape (5,) matching the example in the issue
    # In the original code, inputs are vectors of length 5.
    # Using uniform random values for generality.
    return tf.random.uniform((5,), dtype=tf.float32)

