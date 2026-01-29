# tf.random.uniform((None, 7), dtype=tf.float32) ← Input shape inferred from issue's InputLayer(input_shape=[7])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # In the original issue, reusing the same layer instance in a Sequential model causes an error.
        # The recommended solution is to use Functional API or unique layer instances.
        # Here, we create two separate Dense layers with the same configuration to emulate reuse without error.
        self.dense1 = tf.keras.layers.Dense(units=20, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=20, activation='relu')

    def call(self, inputs):
        # The original problem is about the error when adding the same layer object twice,
        # so here we apply both layers sequentially as in a normal network,
        # but using two different layer instances.
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel with two separate Dense layers,
    # which is the correct way to have repeated layers without error.
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected input shape of [None, 7].
    # Using None for batch size means variable batch size, so we pick an example batch size (e.g., 4).
    return tf.random.uniform((4, 7), dtype=tf.float32)

