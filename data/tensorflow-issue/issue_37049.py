# tf.TensorSpec((None, ), dtype=tf.int64), tf.RaggedTensorSpec((None, None), dtype=tf.string)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # This model expects two inputs:
        #   1) a dense 1D int64 tensor of shape (None,)
        #   2) a ragged 2D string tensor of shape (None, None)

    @tf.function(input_signature=[
        tf.TensorSpec((None,), dtype=tf.int64),
        tf.RaggedTensorSpec((None, None), dtype=tf.string)
    ])
    def call(self, dense, ragged):
        # Normally, direct RaggedTensor inputs to tf.function input_signature 
        # fail when exporting for TF Serving, because SavedModel splits the ragged 
        # tensor into its values and row_splits parts as separate inputs.
        #
        # This function simulates the workaround:
        # Rebuild ragged tensor inside model from inputs, or just process as is.
        # Here, since this is a demo, return a fixed string tensor.
        
        # For demonstration, just return a simple string tensor as in the original code
        return tf.constant(["foobar"], dtype=tf.string)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a tuple of inputs that matches the model input_signature:
    # 1) dense tensor of shape (batch,), dtype int64
    # 2) ragged tensor of shape (batch, None), dtype string
    
    dense = tf.constant([1, 2, 3], dtype=tf.int64)  # shape (3,)
    ragged = tf.ragged.constant([["foo"], ["foo", "bar"], []], dtype=tf.string)  # shape (3, None)
    return (dense, ragged)

