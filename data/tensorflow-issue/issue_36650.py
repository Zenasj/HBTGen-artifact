# tf.random.uniform((batch_size, input_dim)) ← inferred input shape: 2D tensor with shape (batch_size, input_dim)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, units=32):
        super(MyModel, self).__init__()
        self.units = units
        # We use name_scopes around each weight to ensure unique variable names by default,
        # addressing the original issue of identical default weight names.
        with tf.name_scope("w_scope"):
            self.w = self.add_weight(
                name="w",
                shape=(None, self.units),  # Note: shape will be fixed on build
                initializer="random_normal",
                trainable=True,
                dtype=tf.float32)
        with tf.name_scope("b_scope"):
            self.b = self.add_weight(
                name="b",
                shape=(self.units,),
                initializer="random_normal",
                trainable=True,
                dtype=tf.float32)

    def build(self, input_shape):
        # input_shape is expected to be (batch_size, input_dim)
        input_dim = input_shape[-1]
        # Recreate weights with known input dimension
        # This is required because the shape for w was (None, units) in __init__, so override here
        self.w = self.add_weight(
            name="w",
            shape=(input_dim, self.units),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float32)
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float32)
        super(MyModel, self).build(input_shape)

    def call(self, inputs):
        # Simple linear layer: inputs @ w + b
        return tf.matmul(inputs, self.w) + self.b

def my_model_function():
    # Create and return an instance of MyModel initialized with default 32 units
    return MyModel(units=32)

def GetInput():
    # Return a random input tensor matching the expected input shape of MyModel
    # Since units=32, input shape is (batch_size, input_dim)
    # We'll choose batch_size=4 and input_dim=128 as a reasonable example
    batch_size = 4
    input_dim = 128
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

