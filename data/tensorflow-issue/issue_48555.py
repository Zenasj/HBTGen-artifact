# tf.random.uniform((15, 20, 2), dtype=tf.float32)  # inferred input shape from minimal repro (batch=15, time_step=20, n_classes=2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The minimal repro model just reshapes input to (20, 2), but input is already in that shape.
        # To simulate a meaningful model, we'll just apply an identity layer (or a slight transformation).
        # We'll keep it minimal, matching original behavior: outputs shape (batch, 20, 2).
        self.reshape = tf.keras.layers.Reshape((20, 2))

    def call(self, inputs, training=False):
        # Simply reshape input (no trainable params)
        x = self.reshape(inputs)
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random input tensor with shape (15, 20, 2)
    # The original example's input xs corresponds to one hot vector of class '1' reshaped.
    # We'll replicate a similar input shape and dtype.
    batch = 15
    time_steps = 20
    num_classes = 2

    # For x, one hot indices with class 1 repeated (shape 15*20=300, one_hot shape (300,2), reshape to (15, 20, 2))
    indices = tf.ones(batch*time_steps, dtype=tf.int32)
    one_hot = tf.one_hot(indices, depth=num_classes)  # shape (300, 2)
    x = tf.reshape(one_hot, (batch, time_steps, num_classes))  # (15,20,2)
    return x

