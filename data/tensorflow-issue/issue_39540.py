# tf.random.uniform((B, H, W, C), dtype=...) ← input shape is flexible, typical example: (10, 1) for Dense layer input

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model reproduces the original example from the issue about weight creation.
    It has a Dense layer created in __init__ or build, and a dummy trainable variable.
    It shows how to correctly implement build() so that layer weights are registered properly.

    Notes on the issue context:
    - Creating a layer inside build() without calling super().build() or calling layer.build()
      and then querying variables on the model causes issues.
    - The recommended pattern is to create layers in __init__, and call super().build(input_shape) in build().
    - Variables created by self.l1 layer are properly tracked only when the layer is a submodule of the model.
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # Correctly create the Dense layer in __init__, so its weights are tracked
        self.l1 = tf.keras.layers.Dense(5)

    def build(self, input_shape):
        # Call super().build() to set built flag and track the network properly
        super().build(input_shape)
        # Create an additional variable for demonstration, must be created during or after build,
        # so that it is tracked as part of model.variables
        self.dummy = tf.Variable(
            trainable=True,
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(1,), dtype=tf.float32),
            name="dummy_var")
        # Optional: print to confirm variables exist
        # print("Variables after build:", self.variables)

    def call(self, x):
        return self.l1(x)


def my_model_function():
    # Returns an instance of MyModel ready to be built (build will be called on first call)
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the expected input of MyModel
    # Based on example Dense layer input shapes, use batch size 10, input dim 1
    return tf.random.uniform((10, 1), dtype=tf.float32)

