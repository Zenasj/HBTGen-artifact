# tf.random.uniform((None, 20), dtype=tf.float32) <- inferred input shape from example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model demonstrates the pattern discussed in the issue:
        # Can't use model.layers.pop() to remove layers in tf.keras.
        # Instead, define a model with outputs from an intermediate layer.
        # We build a simple feedforward model like the example:
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(20,), name="input")
        # Original layers:
        self.dense1 = tf.keras.layers.Dense(100, activation='relu', name="dense1")
        self.dense2 = tf.keras.layers.Dense(10, activation='relu', name="dense2")
        # New layers added after "removing" last layer:
        self.dense3 = tf.keras.layers.Dense(120, activation='relu', name="dense3")
        self.dense4 = tf.keras.layers.Dense(5, activation='softmax', name="dense4")

    def call(self, inputs):
        # Equivalent to recreating model with some layers "removed"
        # Here, we do NOT use pop but explicitly control the flow
        x = self.input_layer(inputs)
        x = self.dense1(x)
        # Instead of going to dense2 (last layer in orig model),
        # we branch from dense1 output and add new dense3 and dense4:
        x = self.dense3(x)
        out = self.dense4(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor of shape (batch_size, features) matching input shape
    # Batch size: 4 arbitrarily chosen for example
    return tf.random.uniform((4, 20), dtype=tf.float32)

