# tf.random.uniform((B, 4), dtype=tf.float32)  ← Input shape inferred from minimal reproduced example: batch size B, feature dim 4

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple MLP model similar to example:
        self.dense1 = tf.keras.layers.Dense(2, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    @tf.function(jit_compile=True)  # Enable XLA compilation for forward pass
    def call(self, inputs, training=False):
        # Use functional style as in example
        x = self.dense1(inputs)
        out = self.dense2(x)
        return out

    def predict_fast(self, inputs):
        """
        A "fast prediction" method to avoid the overhead of Dataset creation inside model.predict.
        This calls the model directly on inputs with training=False, returning the Tensor output.
        According to issue discussion, this is the recommended pattern for small batches.
        """
        return self.call(inputs, training=False)


def my_model_function():
    """
    Return an instance of MyModel.

    This model replicates the minimal example from the issue causing slowdown on model.predict after compile().
    The compiled call is XLA-jitted for best performance.
    """
    model = MyModel()
    # To match the issue setup: compile the model with optimizer and loss
    model.compile(optimizer='adam', loss='binary_crossentropy', run_eagerly=False)
    # Normally init weights by calling once:
    dummy_input = tf.random.uniform((1, 4), dtype=tf.float32)
    model(dummy_input)
    return model


def GetInput():
    """
    Returns a random input tensor of shape (32, 4), matching the original example batch size and feature dims.
    Use float32 dtype consistent with the typical TensorFlow default.
    """
    return tf.random.uniform((32, 4), dtype=tf.float32)

