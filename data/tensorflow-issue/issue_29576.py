# tf.random.uniform((32, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A demonstration model encapsulating the differing layer build/call behaviors regarding variable naming scopes.
    This MyModel internally creates three Dense layers to illustrate the variable naming inconsistencies described:
    
    - layer_built: explicitly built via build() call inside a name_scope
    - layer_tensor_called: called directly on a zero tensor inside a name_scope
    - layer_input_called: called directly on a Keras Input tensor inside a name_scope
    
    Forward outputs a tuple of the outputs of all three layers on the input.
    """

    def __init__(self):
        super().__init__()
        # Create three dense layers with distinct names
        self.layer_built = tf.keras.layers.Dense(2, name="built")
        self.layer_tensor_called = tf.keras.layers.Dense(2, name="tensor_called")
        self.layer_input_called = tf.keras.layers.Dense(2, name="input_called")

        # We build layer_built explicitly with the appropriate name scopes to reflect the behavior observed
        with tf.name_scope("parent"):
            with tf.name_scope(self.layer_built.name):
                self.layer_built.build((None, 3))  # Input shape fixed as 3 per examples

    def call(self, inputs, training=None):
        # Simulate the usages inside a "parent" name scope to observe variable naming effects

        # The layer_tensor_called is called on a tensor inside a name scope "parent" to mimic experiment context
        with tf.name_scope("parent"):
            out_tensor_called = self.layer_tensor_called(tf.zeros_like(inputs))

        # The layer_input_called is called directly on the Keras Input tensor inside the same parent scope
        # According to the issue, this call ignores the parent scope for variable naming
        with tf.name_scope("parent"):
            out_input_called = self.layer_input_called(inputs)

        # layer_built is already built; call it directly on inputs (no name scope applied here for call)
        out_built = self.layer_built(inputs)

        # Return all outputs to verify and compare
        return out_built, out_tensor_called, out_input_called

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input of shape (32, 3) matching expected input dimension
    # Using uniform random values, dtype float32 as typical for Keras Dense layers
    return tf.random.uniform((32, 3), dtype=tf.float32)

