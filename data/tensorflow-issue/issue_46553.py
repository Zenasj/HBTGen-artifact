# tf.random.uniform((B,), dtype=tf.string) ← inferred input shape is scalar input, output is string vector of length `length`

import tensorflow as tf

def initializer(shape, dtype=None):
    # Initialize a string tensor with values ['0', '1', '2', ..., str(shape[0]-1)]
    # Note: This initializer returns a tf.string tensor of shape `shape`.
    string_values = [str(x) for x in range(shape[0])]
    return tf.constant(string_values, dtype=tf.string)

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, length=10, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.length = length

    def build(self, input_shape=None):
        # Add keys as a variable of dtype string, not trainable
        # Using add_weight with dtype=tf.string triggers issues on set_weights
        # But kept to replicate original logic; see comments below.
        self.keys = self.add_weight(
            shape=(self.length,),
            initializer=initializer,
            dtype=tf.string,
            trainable=False,
            name="keys"
        )
        self.built = True

    def call(self, x):
        # Simply return the keys string weights regardless of input
        return self.keys

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({'length': self.length})
        return config

class MyModel(tf.keras.Model):
    def __init__(self, length=10):
        super(MyModel, self).__init__()
        # Two approaches to maintain string data as "weights":
        # 1) Use add_weight with dtype string (problematic for set_weights).
        # 2) Use tf.Variable for string storage (recommended workaround).
        # We will use CustomLayer (with add_weight) and an alternative layer using tf.Variable
        self.length = length
        self.lookup_add_weight = CustomLayer(length=length, name="lookup_add_weight")

        # Alternative sublayer that uses tf.Variable for strings, non-trainable
        self.lookup_variable = tf.keras.layers.Layer(name="lookup_variable")
        self.lookup_variable.keys_var = tf.Variable(
            [str(i) for i in range(length)],
            dtype=tf.string,
            trainable=False,
            shape=(length,)
        )

    def call(self, inputs):
        # Get outputs from both layers
        out_add_weight = self.lookup_add_weight(inputs)  # from add_weight (this is string tensor)
        # from tf.Variable storing strings
        out_var = self.lookup_variable.keys_var

        # Compare outputs by elementwise equality (dtype string supported for tf.equal)
        # This will yield a boolean tensor of shape (length,)
        comparison = tf.equal(out_add_weight, out_var)

        # For demonstration return a dictionary of outputs
        return {
            "add_weight_strings": out_add_weight,
            "variable_strings": out_var,
            "are_equal": comparison,
            # Additionally return if all strings are equal as a single boolean scalar
            "all_equal": tf.reduce_all(comparison)
        }

def my_model_function():
    # Returns an instance of MyModel with default length=10
    return MyModel(length=10)

def GetInput():
    # Return a scalar dummy input compatible with MyModel call
    # Input shape () scalar tensor - matches Input(shape=()) usage in example
    return tf.constant(0)

