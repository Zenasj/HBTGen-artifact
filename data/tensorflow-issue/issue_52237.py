# tf.random.uniform((B, 1, 4), dtype=tf.float32)
import tensorflow as tf
from tensorflow.python.ops.gen_cudnn_rnn_ops import cudnn_rnn
from tensorflow.python.keras.layers.core import TFOpLambda
from tensorflow.python.util.dispatch import add_dispatch_support

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Wrapping cudnn_rnn with dispatch support works correctly with KerasTensor inputs
        self.cudnn_rnn_dispatch = add_dispatch_support(cudnn_rnn)
        # Using TFOpLambda directly wraps the op as a layer but causes issues with functional API
        self.cudnn_layer = TFOpLambda(cudnn_rnn)

    def call(self, inputs, training=False):
        # inputs is expected to be a tensor of shape (batch, 1, 4)
        # Demonstrate both approaches and compare outputs
        # Note: cudnn_rnn expects input and hx tensors; we use the input for both input and hx
        batch_size = tf.shape(inputs)[0]
        # Initial hidden state as zeros matching expected size for GRU (hidden size=128)
        zeros = tf.zeros([batch_size, 128], dtype=inputs.dtype)

        # Use the dispatched cudnn_rnn first -- this should work correctly with KerasTensor
        out_dispatch = self.cudnn_rnn_dispatch(inputs, inputs, 0, zeros, rnn_mode='gru')[0]

        # Use the TFOpLambda layer directly -- may error during functional construction,
        # but in eager mode this will execute (per the issue description)
        try:
            out_layer = self.cudnn_layer(inputs, inputs, 0, zeros, rnn_mode='gru')[0]
        except TypeError:
            # If it fails, fallback to zeros to avoid failing model call
            out_layer = tf.zeros_like(out_dispatch)

        # Compare outputs numerically, allowing some tolerance as cudnn_rnn is GPU kernel
        diff = tf.abs(out_dispatch - out_layer)
        are_close = tf.reduce_all(diff < 1e-5)

        # Return a tuple: (dispatched output, direct layer output, bool tensor if close)
        return (out_dispatch, out_layer, are_close)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random uniform input tensor of shape (batch_size=2, 1, 4)
    # dtype float32 to match cudnn_rnn requirements
    return tf.random.uniform((2, 1, 4), dtype=tf.float32)

