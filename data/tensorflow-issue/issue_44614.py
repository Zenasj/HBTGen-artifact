# tf.random.uniform((B, 1), dtype=tf.float32)  ← The original input shape is (None, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer with kernel initialized to constant 1, linear activation
        self.dense = tf.keras.layers.Dense(1, activation="linear",
                                           kernel_initializer=tf.keras.initializers.Constant(1),
                                           name="dense")
        # Dropout layer with fixed seed=12 and dropout rate 0.5
        self.dropout = tf.keras.layers.Dropout(0.5, seed=12, name="dropout")
        
    def call(self, inputs, training=True):
        # Apply dense layer
        x = self.dense(inputs)
        # Apply dropout with training flag set according to call argument
        x = self.dropout(x, training=training)
        return x

def my_model_function():
    # Return a new instance of MyModel with specified weights/initialization
    return MyModel()

def GetInput():
    # Return a random tensor with shape (1, 1), dtype float32 matching model input shape
    # Using uniform random values to simulate a sample input
    return tf.random.uniform((1, 1), dtype=tf.float32)

