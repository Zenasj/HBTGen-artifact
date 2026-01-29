# tf.zeros(shape=(3, 10), dtype=tf.float32) ← The example input shape used in issue to demonstrate BatchNormalization behavior

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Replicating a minimal BatchNormalization usage scenario as in the issue
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        # The BatchNormalization layer's updates are internal to the layer in tf.keras.
        # They are not added to UPDATE_OPS collection globally.
        # Instead, they need to be run explicitly if using custom training loops.
        # Here, we just return the normalized output.
        return self.bn(inputs, training=training)

def my_model_function():
    # Return new instance of MyModel
    return MyModel()

def GetInput():
    # Create a tensor matching the example input in the issue
    # Shape (3, 10), dtype float32, all zeros as in example code
    return tf.zeros(shape=(3, 10), dtype=tf.float32)

