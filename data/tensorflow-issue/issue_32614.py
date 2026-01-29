# tf.random.uniform((256, 1), dtype=tf.float32) ← input shape inferred from batch_input_shape in Dense layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Equivalent to tf.keras.models.Sequential([
        #    tf.keras.layers.Dense(16, batch_input_shape=[256,1], activation='relu'),
        #    tf.keras.layers.Dense(16, activation='tanh'),
        #    tf.keras.layers.Dense(1)
        # ])
        # Note: batch_input_shape includes batch size 256, so input shape is (256, 1)
        # For flexibility, we still keep batch dimension dynamic, but define layers same.
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Initialize variables as in the original code
    # The original had lr and reg_param as tf.Variable, we include them here as attributes
    model = MyModel()
    # Initialize lr and reg_param as variables (default values are 1.0)
    model.lr = tf.Variable(1., trainable=False, dtype=tf.float32)
    model.reg_param = tf.Variable(1., trainable=False, dtype=tf.float32)
    return model

def GetInput():
    # Return a random tensor matching input shape expected by MyModel
    # Input shape from batch_input_shape is (256, 1)
    # dtype inferred as default tf.float32
    return tf.random.uniform((256, 1), dtype=tf.float32)

