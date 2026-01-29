# tf.random.uniform((B, 1), dtype=tf.float32) ← Here input shape is (?, 1) as per the original Input layer shape

import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(MyDenseLayer, self).__init__()
        self.units = units
        # Initialize Dense layer here instead of build to properly track variables
        self.dense = tf.keras.layers.Dense(units=self.units)

    def call(self, inputs, mask=None):
        return self.dense(inputs)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Instead of wrapping a Layer inside the call, instantiate it as a sublayer to ensure variables are tracked and checkpointed.
        self.my_dense = MyDenseLayer(units=1)

    def call(self, inputs, training=False):
        # Forward pass through the encapsulated MyDenseLayer
        return self.my_dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    # Users will have a trackable model whose variables can be checkpointed/restored properly
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected input shape
    # Batch size is arbitrarily 4 here; height=1, width not applicable, channels=not used (1D tensor)
    return tf.random.uniform((4, 1), dtype=tf.float32)

