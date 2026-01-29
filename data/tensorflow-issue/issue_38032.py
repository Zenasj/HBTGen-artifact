# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from Input(shape=(10,)) in the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define two heads as per the example: one Dense(5), another Dense(2)
        self.dense1 = tf.keras.layers.Dense(5, name='output_1')
        self.dense2 = tf.keras.layers.Dense(2, name='output_2')

    def call(self, inputs, training=False):
        # Forward pass producing two outputs
        x1 = self.dense1(inputs)
        x2 = self.dense2(inputs)
        # Return outputs as a tuple, since model.fit() supports tuple outputs better than list for multi-output models
        return (x1, x2)

def my_model_function():
    # Return an instance of MyModel; weights are randomly initialized by default
    return MyModel()

def GetInput():
    # Generate a batch of one random input tensor with shape (1, 10) matching the example's Input(shape=(10,))
    return tf.random.uniform((1, 10), dtype=tf.float32)

