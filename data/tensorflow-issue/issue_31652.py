# tf.random.uniform((B, 3), dtype=tf.float32) ← Input shape is (batch_size, 3) for three input variables normalized between 0 and 1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the weights as trainable variables, initialized similar to the original code's constants
        # Input layer has 3 features
        # Hidden layer with two nodes (x4 and x5), weighted sums from inputs
        self.w0 = tf.Variable(0.1, dtype=tf.float32)
        self.w1 = tf.Variable(0.1, dtype=tf.float32)
        self.w2 = tf.Variable(0.1, dtype=tf.float32)
        self.w3 = tf.Variable(0.1, dtype=tf.float32)
        # Output layer weights from hidden nodes to single output
        self.w4 = tf.Variable(0.1, dtype=tf.float32)
        self.w5 = tf.Variable(0.1, dtype=tf.float32)

    def call(self, inputs, training=None):
        # inputs: shape (batch_size, 3), values expected normalized [0,1]

        # Hidden layer computations per original logic:
        # x4 = (x1*w0) + (x2*w1)
        # x5 = (x2*w2) + (x3*w3)
        x1 = inputs[:, 0]
        x2 = inputs[:, 1]
        x3 = inputs[:, 2]

        x4 = x1 * self.w0 + x2 * self.w1
        x5 = x2 * self.w2 + x3 * self.w3

        # Output layer weighted sum:
        x6 = x4 * self.w4 + x5 * self.w5  # Shape (batch_size,)

        # Decision rule:
        # If x6 > 0.5 output 1 else -1, as in original code
        y_pred = tf.where(x6 > 0.5, 1.0, -1.0)

        return y_pred

def my_model_function():
    # Return an instance of MyModel (untrained weights start at 0.1)
    return MyModel()

def GetInput(batch_size=4):
    # Return a random tensor input matching the input expected by MyModel:
    # shape (batch_size, 3) with values normalized between 0 and 1
    return tf.random.uniform((batch_size, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

