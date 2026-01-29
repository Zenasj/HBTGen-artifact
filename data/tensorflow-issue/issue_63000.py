# tf.random.uniform((1, 2, 2, 3), dtype=tf.float32)

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We want to replicate the reported issue scenario:
        # Adding a constant tensor to the input using a Keras Add layer.
        #
        # According to the issue, Add layer of a constant tensor directly
        # caused problems when saving/loading the model.
        #
        # To avoid that problem, the best practice is to use a Lambda layer
        # to add the constant instead of direct Add layer with constant input.
        #
        # However, since the original issue reproduces the bug with Add,
        # here we implement it exactly as in the code for demonstration.
        #
        # Note: For practical usage, one would use Lambda or tf.add inside call.
        
        self.add_layer = tf.keras.layers.Add()

        # Constant tensor to be added: shape (1, 1, 1, 3), broadcastable to input
        self.constant = tf.constant(1.0, shape=(1, 1, 1, 3))

    def call(self, inputs):
        # inputs shape: (batch, 2, 2, 3)
        # We wrap the constant as a tensor to feed Add layer
        # The Add layer expects a list of tensors.
        # This matches original example:
        # tf.keras.layers.Add()([input_tensor, const_tensor])
        
        # We must expand dims for constant to match batch-size, but broadcasting should handle it.
        const_batch = tf.broadcast_to(self.constant, tf.shape(inputs))
        return self.add_layer([inputs, const_batch])


def my_model_function():
    # Returns an instance of MyModel.
    return MyModel()


def GetInput():
    # Generate a random input tensor matching the expected input shape: (1, 2, 2, 3)
    # Using uniform distribution for demonstration
    return tf.random.uniform((1, 2, 2, 3), dtype=tf.float32)

