# tf.random.uniform((32, 32, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model encapsulates a simple Dense layer applied to an input tensor.
    Since the issue discussed relates to problems caused when storing lists as attributes
    in tf.keras.Model subclasses (where lists become ListWrapper instances),
    this example avoids storing plain Python lists on self, instead works directly
    with tensors to prevent the indexing/serialization issues described.
    
    The call method applies a Dense layer of output size 3 to the input.

    Assumptions:
    - Input shape (32, 32, 1) inferred from the example.
    - Dense layer requires input shape (32, 32, 1), outputs shape (32, 32, 3).
    - The forward logic avoids storing lists on `self` to not trigger ListWrapper behavior.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(3)

        # Build the dense layer with fixed input shape for clarity
        self.dense.build((32, 32, 1))

    def call(self, inputs, training=False):
        # inputs expected shape: (32, 32, 1), a single tensor (no list)
        # Applies dense layer
        return self.dense(inputs)


def my_model_function():
    # Returns an instance of MyModel
    return MyModel()


def GetInput():
    # Returns a random tensor input matching the expected input shape (32, 32, 1)
    # dtype float32 is default for tf.random.uniform
    return tf.random.uniform((32, 32, 1), dtype=tf.float32)

