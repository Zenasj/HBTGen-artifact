# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from example Input(shape=(10,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer with trainable=True as per example
        self.trainable_layer = tf.keras.layers.Dense(15, trainable=True, name="trainable_layer")
        # Dense layer with trainable=False as per example
        self.nontrainable_layer = tf.keras.layers.Dense(5, trainable=False, name="nontrainable_layer")

    def call(self, inputs):
        x = self.trainable_layer(inputs)
        output = self.nontrainable_layer(x)
        return output

def my_model_function():
    # Return an instance of MyModel as defined above
    return MyModel()

def GetInput():
    # Return a random tensor input that matches MyModel's expected input: shape (batch, 10)
    # Batch size chosen arbitrarily as 4
    return tf.random.uniform((4, 10), dtype=tf.float32)

