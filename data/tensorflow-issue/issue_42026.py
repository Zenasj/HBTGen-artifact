# tf.random.uniform((B, 5), dtype=tf.float32) ← Input shape inferred as (batch_size, 5) from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        # Instead of creating layers as attributes before super().__init__,
        # define layers locally and build Functional model by calling super with inputs and outputs,
        # per recommended pattern in the issue comments.
        
        dense_layer = tf.keras.layers.Dense(10)
        inputs = tf.keras.Input(shape=(5,))
        outputs = dense_layer(inputs)
        
        # Initialize the base Model with the Functional API model constructed
        super(MyModel, self).__init__(inputs=inputs, outputs=outputs, **kwargs)
        
        # The layer tracking and call method will be managed automatically by the base Model
        # since outputs depend on inputs and layers included.
        
        # Print summary to verify input/output shapes (optional)
        self.summary()

    def call(self, inputs):
        # As this instance is constructed from Functional API inputs/outputs,
        # the call method is automatically managed,
        # but we override to maintain compatibility if needed.
        return super(MyModel, self).call(inputs)


def my_model_function():
    # Return an instance of MyModel; newly constructed with the pattern
    return MyModel()


def GetInput():
    # Return a random tensor input matching shape (batch_size, 5)
    # batch_size arbitrarily chosen (e.g., 4)
    batch_size = 4
    return tf.random.uniform((batch_size, 5), dtype=tf.float32)

