# tf.random.uniform((B, input_dim), dtype=tf.float32) ← Assuming input is a batch of vectors for Dense layer

import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class SumConstraint(Constraint):
    """
    Custom constraint to enforce weights to be non-negative and
    each vector along the specified axis sums to 1.
    
    Note:
    - The original user code tried to convert tensor to numpy inside graph which is invalid in TF2.
    - Here we implement this purely with TensorFlow ops so it works with graph modes and XLA.
    - We add a small epsilon to avoid division by zero.
    """
    def __init__(self, axis=0):
        self.axis = axis
        self.epsilon = 1e-12

    def __call__(self, w):
        # Clip to non-negative
        w = tf.nn.relu(w)

        # Compute sum along axis, keep dims for broadcast
        sum_axis = tf.reduce_sum(w, axis=self.axis, keepdims=True)

        # Normalize weights so sum along axis is 1 (avoid division by zero)
        w = w / (sum_axis + self.epsilon)
        return w

    def get_config(self):
        return {'axis': self.axis}

class MyModel(tf.keras.Model):
    """
    A simple Keras model with a Dense layer that uses the custom SumConstraint.
    This illustrates how to properly apply a custom constraint in TF2.

    The input shape is (batch_size, input_dim).
    Output is (batch_size, 3) for demonstration.
    """
    def __init__(self, input_dim=6, units=3):
        super(MyModel, self).__init__()
        # Dense layer with kernel constrained to be non-negative and sum to 1 along axis=0 (columns)
        # axis=0 means sum of weights across input_dim for each output unit sums to 1
        self.dense = Dense(
            units,
            activation=None,
            kernel_constraint=SumConstraint(axis=0),
            input_shape=(input_dim,)
        )

    def call(self, inputs):
        # Forward pass through Dense layer
        return self.dense(inputs)

def my_model_function():
    # Create and return an instance of MyModel with default input_dim=6
    # Note: model weights are initialized randomly; no pretrained weights loaded
    return MyModel()

def GetInput():
    # Return a random tensor input that matches MyModel input shape
    # Batch size is arbitrary, here chosen 4 for test
    # input_dim=6 matches MyModel's expected input shape
    batch_size = 4
    input_dim = 6
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

