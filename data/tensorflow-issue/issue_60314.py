# tf.constant shape: (1, 2), dtype=tf.float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model encapsulates two approaches to compute softmax and 
    categorical cross-entropy loss from the issue description:

    1) Using Dense layer with `softmax` activation directly.
    2) Using Dense layer with `linear` activation followed by explicit Softmax layer.

    The forward method returns the difference between the gradients 
    of the loss wrt the inputs for these two approaches to highlight 
    the issue described: different backprop gradients on zero label vector.

    This fusion allows for direct comparison as discussed in the issue.
    """

    def __init__(self):
        super().__init__()
        # Dense layer with softmax activation in one step
        self.dense_softmax = tf.keras.layers.Dense(2, activation="softmax", use_bias=False)
        # Dense layer linear + explicit Softmax layer
        self.dense_linear = tf.keras.layers.Dense(2, activation="linear", use_bias=False)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x):
        y = tf.zeros_like(x)  # labels = all zeros as in the issue

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            # Approach 1: Dense with softmax activation
            out1 = self.dense_softmax(x)
            loss1 = tf.keras.losses.categorical_crossentropy(y, out1)

            # Approach 2: Dense linear + Softmax layers
            lin = self.dense_linear(x)
            out2 = self.softmax(lin)
            loss2 = tf.keras.losses.categorical_crossentropy(y, out2)

        # Calculate gradients with respect to input x for both approaches
        grad1 = tape.gradient(loss1, x)
        grad2 = tape.gradient(loss2, x)

        # Return the difference of gradients to expose the gradient discrepancy issue
        return grad1 - grad2

def my_model_function():
    return MyModel()

def GetInput():
    # Based on issue example:
    # Input shape: (1, 2), values arbitrary but same as issue input
    # dtype float32
    return tf.constant([[0.1, 0.90]], dtype=tf.float32)

