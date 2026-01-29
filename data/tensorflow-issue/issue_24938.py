# tf.random.uniform((B, H, W, C), dtype=tf.float32) - Assumed input shape for demonstration: 
# batch size B, height H, width W, channels C compatible with concatenation on dims 1 and 2

import tensorflow as tf
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, x):
        """
        Implements the logic analogous to the ReshapeLayer described in the issue.
        Takes input tensor x of shape [B, H, W, C].
        Concatenates a zero tensor of shape [B, 1, W, C] on dimension 1,
        then another zero tensor of shape [B, H+1, 1, C] on dimension 2.
        
        Uses tf.zeros directly instead of K.zeros to avoid issues tracking tensors as variables in graph mode.
        """
        s = tf.shape(x)  # dynamic shape: [B, H, W, C]
        
        # zeros_w corresponds to appending a zero "row" along dimension 1 (height dim)
        zeros_w = tf.zeros((s[0], 1, s[2], s[3]), dtype=x.dtype)  # shape [B,1,W,C]
        r = tf.concat([x, zeros_w], axis=1)  # shape [B, H+1, W, C]
        
        s_r = tf.shape(r)  # shape [B, H+1, W, C]
        # zeros_h corresponds to appending a zero "column" along dimension 2 (width dim)
        zeros_h = tf.zeros((s_r[0], s_r[1], 1, s_r[3]), dtype=x.dtype)  # shape [B,H+1,1,C]
        r = tf.concat([r, zeros_h], axis=2)  # shape [B, H+1, W+1, C]
        
        return r

def my_model_function():
    # Returns an instance of MyModel, no special initialization needed.
    return MyModel()

def GetInput():
    """
    Returns a random tensor with the expected input shape:
    We assume B=2, H=4, W=5, C=3 as illustrative example dimensions.
    This matches the usage in MyModel where concatenation on dims 1 and 2 occurs.
    """
    B, H, W, C = 2, 4, 5, 3
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)

