# tf.random.uniform((2, 256, 256, 1), dtype=tf.float32) ← input shape inferred from usage in main

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, ind1, w1):
        """
        ind1: tf.Tensor of shape [367, 217, 721, 2], dtype tf.int32 (indices)
        w1: tf.Tensor of shape [367, 217, 721, 1, 1], dtype tf.float32 (weights)
        """
        super(MyModel, self).__init__()
        # Store the indices and weights as tf.Tensors to avoid embedding large numpy arrays literally in graph
        self.ind1 = ind1
        self.w1 = tf.cast(w1, tf.float32)
        # Trainable scalar parameter
        self.lambda1 = tf.Variable(initial_value=0.1, trainable=True, name='lambda1')

    def call(self, inputs, training=False):
        """
        inputs: Tensor of shape [B, H, W, C], here [2,256,256,1]
        Returns: Tensor shaped roughly [B,217,721,1] after gather, weighted sum and scaling
        """
        y = inputs
        # Repeatedly transform and gather on y tensor
        for _ in range(5):
            # Transpose to bring batch dimension to last axis for gather_nd
            y = tf.transpose(y, [1, 2, 3, 0])  # shape [H, W, C, B]
            # Gather slices along last axis with indices ind1.
            # ind1 shape [367, 217, 721, 2] selects elements from last two dims of y (which are C and B)
            # Here we multiply y by 1.0 to ensure float type and compatibility
            gathered = tf.gather_nd(y * 1.0, self.ind1)  # gathered shape [367, 217, 721]
            # Apply the weights element-wise (broadcasted)
            weighted = gathered * self.w1  # shape preserved [367,217,721,1,1]
            # Reduce sum over first dimension (dim=0), collapsing dim 367
            y = tf.reduce_sum(weighted, axis=0)  # shape [217, 721, 1, 1]
            # Transpose back to original order with batch as first axis (approximation)
            y = tf.transpose(y, [3, 0, 1, 2])  # shape [1,217,721,1]
            # Apply learned scalar lambda1 multiplier
            y = self.lambda1 * y
        return y


def my_model_function():
    # To avoid RAM exhaustion by embedding huge numpy arrays as graph constants,
    # we generate tensors directly here.
    # Shapes and value ranges are based on issue context:
    # ind1: int indices in range [0, 300), shape [367,217,721,2]
    # w1: normal random weights, shape [367,217,721,1,1]

    ind1 = tf.random.uniform(
        shape=[367, 217, 721, 2],
        minval=0,
        maxval=300,
        dtype=tf.int32
    )
    w1 = tf.random.normal(shape=[367, 217, 721, 1, 1], dtype=tf.float32)
    return MyModel(ind1, w1)


def GetInput():
    # Produces a batch of 2 random images of shape 256x256 with 1 channel,
    # matching input shape used in example
    return tf.random.uniform(shape=[2, 256, 256, 1], dtype=tf.float32)

