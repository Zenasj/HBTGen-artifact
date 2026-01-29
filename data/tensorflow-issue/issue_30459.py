# tf.random.uniform((B, 1), dtype=tf.float64) ← The input is a batch of sequences with shape (batch_size, 1), dtype=tf.float64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # No trainable weights are defined; the model computes tf.linalg.expm of inputs' gram matrix
        
    @tf.function
    def call(self, inputs, training=None):
        """
        inputs: Tensor of shape (batch_size, 1), dtype=tf.float64 (or compatible float dtype)
        
        The model calculates X = inputs^T @ inputs (gram matrix), then outputs matrix exponential expm(X).
        """
        # Compute gram matrix of inputs: shape (1, 1) if inputs shape is (batch_size, 1)
        # Because inputs is (batch_size, 1), tf.transpose(inputs) is (1, batch_size)
        X = tf.matmul(tf.transpose(inputs), inputs)
        
        # Compute matrix exponential of X
        out = tf.linalg.expm(X)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    """
    Generate input tensor suitable for MyModel.
    
    - Batch size chosen randomly between 100 and 300 (to mimic the original code's dynamic size).
    - Shape is (batch_size, 1).
    - Use dtype=tf.float64 due to original example.
    """
    batch_size = tf.random.uniform([], minval=100, maxval=300, dtype=tf.int32)
    return tf.ones((batch_size, 1), dtype=tf.float64)

