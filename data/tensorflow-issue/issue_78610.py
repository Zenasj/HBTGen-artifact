# tf.random.uniform((4, 2), dtype=tf.float32) ← inferred input shape from dummy_data_x shape [[0,0],[1,0],[0,1],[1,1]]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with 2 units and softmax, as per original CusModel
        self.dense = tf.keras.layers.Dense(units=2, activation='softmax', name='output')

    def call(self, x):
        return self.dense(x)

def my_model_function():
    # Returns a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Create dummy input tensor matching shape and type used in the original code
    # The original dummy_data_x is shape (4, 2) with values {0,1}.
    # To keep it general, use uniform integers 0 or 1 cast to float32, same shape.
    dummy_input = tf.random.uniform(shape=(4, 2), minval=0, maxval=2, dtype=tf.int32)
    dummy_input = tf.cast(dummy_input, tf.float32)
    return dummy_input

